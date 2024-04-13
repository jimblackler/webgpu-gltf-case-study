import {mat4, vec3} from '../node_modules/gl-matrix/esm/index.js';
import {TinyGltfWebGpu} from './tiny-gltf.js'
import {wgsl} from '../node_modules/wgsl-preprocessor/wgsl-preprocessor.js'

const GltfRootDir = './glTF-Sample-Models/2.0';

const GltfModels = {
  antique_camera: `${GltfRootDir}/AntiqueCamera/glTF/AntiqueCamera.gltf`,
  buggy: `${GltfRootDir}/Buggy/glTF-Binary/Buggy.glb`,
  corset: `${GltfRootDir}/Corset/glTF-Binary/Corset.glb`,
  damaged_helmet: `${GltfRootDir}/DamagedHelmet/glTF-Binary/DamagedHelmet.glb`,
  flight_helmet: `${GltfRootDir}/FlightHelmet/glTF/FlightHelmet.gltf`,
  sponza: `./sponza-optimized/Sponza.gltf`,
  mc_laren: `${GltfRootDir}/McLaren.glb`,
  porsche_gt3_rs: `${GltfRootDir}/porsche_gt3_rs.glb`,
};

// Shader locations and source are unchanged from the previous sample.
const ShaderLocations = {
  POSITION: 0,
  NORMAL: 1,
  // Add texture coordinates to the list of attributes we care about.
  TEXCOORD_0: 2
};

const FRAME_BUFFER_SIZE = Float32Array.BYTES_PER_ELEMENT * 36;

export async function gltfDemo(startup_model) {
  const colorFormat = navigator.gpu?.getPreferredCanvasFormat?.() || 'bgra8unorm';
  const frameArrayBuffer = new ArrayBuffer(FRAME_BUFFER_SIZE);
  const projectionMatrix = new Float32Array(frameArrayBuffer, 0, 16);
  const viewMatrix = new Float32Array(frameArrayBuffer, 16 * Float32Array.BYTES_PER_ELEMENT, 16);
  const canvas = document.querySelector('.webgpu-canvas');

  const context = canvas.getContext('webgpu');

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  context.configure({
    device: device,
    format: colorFormat,
    alphaMode: 'opaque',
  });

  const frameUniformBuffer = device.createBuffer({
    size: FRAME_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const frameBindGroupLayout = device.createBindGroupLayout({
    label: `Frame BindGroupLayout`,
    entries: [{
      binding: 0, // Camera/Frame uniforms
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: {},
    }],
  });

  const frameBindGroup = device.createBindGroup({
    label: `Frame BindGroup`,
    layout: frameBindGroupLayout,
    entries: [{
      binding: 0, // Camera uniforms
      resource: {buffer: frameUniformBuffer},
    }],
  });

  const gltf = await new TinyGltfWebGpu(device).loadFromUrl(GltfModels[startup_model]);
  const sceneAabb = gltf.scenes[gltf.scene].aabb;

  orbitCamera(canvas, sceneAabb.center, sceneAabb.radius * 2.0,
      sceneAabb.radius, sceneAabb.radius * 1.5, mtx => viewMatrix.set(mtx));

  const pipelineGpuData = new Map();
  const shaderModules = new Map();

  const instanceBindGroupLayout = device.createBindGroupLayout({
    label: `glTF Instance BindGroupLayout`,
    entries: [{
      binding: 0, // Node uniforms
      visibility: GPUShaderStage.VERTEX,
      buffer: {type: 'read-only-storage'},
    }],
  });

  const materialBindGroupLayout = device.createBindGroupLayout({
    label: `glTF Material BindGroupLayout`,
    entries: [{
      binding: 0, // Material uniforms
      visibility: GPUShaderStage.FRAGMENT,
      buffer: {},
    }, {
      binding: 1, // Texture sampler
      visibility: GPUShaderStage.FRAGMENT,
      sampler: {},
    }, {
      binding: 2, // BaseColor texture
      visibility: GPUShaderStage.FRAGMENT,
      texture: {},
    }], // Omitting additional material properties for simplicity
  });

  const primitiveInstances = {
    matrices: new Map(),
    total: 0,
    arrayBuffer: null,
    offset: 0,
  };

  for (const node of gltf.nodes) {
    if ('mesh' in node) {
      const mesh = gltf.meshes[node.mesh];
      for (const primitive of mesh.primitives) {
        let instances = primitiveInstances.matrices.get(primitive);
        if (!instances) {
          instances = [];
          primitiveInstances.matrices.set(primitive, instances);
        }
        instances.push(node);
      }
      primitiveInstances.total += mesh.primitives.length;
    }
  }

  const materialGpuData = new Map();
  for (const material of gltf.materials) {
    // Create a uniform buffer for this material and populate it with the material properties.
    const materialUniformBuffer = device.createBuffer({
      // Even though the struct in the shader only uses 5 floats, WebGPU requires buffer
      // bindings to be padded to multiples of 16 bytes, so we're going to allocate a bit
      // extra.
      size: 8 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    const materialBufferArray = new Float32Array(materialUniformBuffer.getMappedRange());
    materialBufferArray.set(material.pbrMetallicRoughness?.baseColorFactor || [1, 1, 1, 1]);
    materialBufferArray[4] = material.alphaCutoff || 0.5;
    materialUniformBuffer.unmap();

    let baseColor = gltf.gpuTextures[material.pbrMetallicRoughness?.baseColorTexture?.index];
    if (!baseColor) {
      throw Error();
    }

    materialGpuData.set(material, {
      bindGroup: device.createBindGroup({
        label: `glTF Material BindGroup`,
        layout: materialBindGroupLayout,
        entries: [{
          binding: 0, // Material uniforms
          resource: {buffer: materialUniformBuffer},
        }, {
          binding: 1, // Sampler
          resource: baseColor.sampler,
        }, {
          binding: 2, // BaseColor
          resource: baseColor.texture.createView(),
        }],
      }),
    });
  }

  // Create a buffer large enough to contain all the instance matrices for the entire scene.
  const instanceBuffer = device.createBuffer({
    size: 32 * Float32Array.BYTES_PER_ELEMENT * primitiveInstances.total,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  primitiveInstances.arrayBuffer = new Float32Array(instanceBuffer.getMappedRange());

  for (const mesh of gltf.meshes) {
    for (const primitive of mesh.primitives) {
      const bufferLayout = new Map();
      const gpuBuffers = new Map();
      let drawCount = 0;

      for (const [attribName, accessorIndex] of Object.entries(primitive.attributes)) {
        const accessor = gltf.accessors[accessorIndex];
        const bufferView = gltf.bufferViews[accessor.bufferView];

        const shaderLocation = ShaderLocations[attribName];
        if (shaderLocation === undefined) {
          continue;
        }

        const offset = accessor.byteOffset;

        let buffer = bufferLayout.get(accessor.bufferView);
        let gpuBuffer;

        let separate = buffer && (Math.abs(offset - buffer.attributes[0].offset) >= buffer.arrayStride);
        if (!buffer || separate) {
          buffer = {
            arrayStride: bufferView.byteStride || TinyGltfWebGpu.packedArrayStrideForAccessor(accessor),
            attributes: [],
          };

          bufferLayout.set(separate ? attribName : accessor.bufferView, buffer);
          gpuBuffers.set(buffer, {
            buffer: gltf.gpuBuffers[accessor.bufferView],
            offset
          });
        } else {
          gpuBuffer = gpuBuffers.get(buffer);
          gpuBuffer.offset = Math.min(gpuBuffer.offset, offset);
        }

        buffer.attributes.push({
          shaderLocation,
          format: TinyGltfWebGpu.gpuFormatForAccessor(accessor),
          offset,
        });

        drawCount = accessor.count;
      }

      for (const buffer of bufferLayout.values()) {
        const gpuBuffer = gpuBuffers.get(buffer);
        for (const attribute of buffer.attributes) {
          attribute.offset -= gpuBuffer.offset;
        }
        // Sort the attributes by shader location.
        buffer.attributes = buffer.attributes.sort((a, b) => a.shaderLocation - b.shaderLocation);
      }
      // Sort the buffers by their first attribute's shader location.
      const sortedBufferLayout = [...bufferLayout.values()].sort(
          (a, b) => a.attributes[0].shaderLocation - b.attributes[0].shaderLocation);

      // Ensure that the gpuBuffers are saved in the same order as the buffer layout.
      const sortedGpuBuffers = sortedBufferLayout.map(buffer => gpuBuffers.get(buffer));

      const instances = primitiveInstances.matrices.get(primitive);

      const first = primitiveInstances.offset;
      const count = instances.length;

      instances.forEach((instance, i) => {
        primitiveInstances.arrayBuffer.set(instance.worldMatrix, (first + i) * 32);
        primitiveInstances.arrayBuffer.set(instance.normalMatrix, (first + i) * 32 + 16);
      });

      primitiveInstances.offset += count;
      const gpuPrimitive = {
        buffers: sortedGpuBuffers,
        drawCount,
        instances: primitiveInstances
      };

      if ('indices' in primitive) {
        const accessor = gltf.accessors[primitive.indices];
        gpuPrimitive.indexBuffer = gltf.gpuBuffers[accessor.bufferView];
        gpuPrimitive.indexOffset = accessor.byteOffset;
        gpuPrimitive.indexType = TinyGltfWebGpu.gpuIndexFormatForComponentType(accessor.componentType);
        gpuPrimitive.drawCount = accessor.count;
      }

      const material = gltf.materials[primitive.material];
      const gpuMaterial = materialGpuData.get(material);

      // Start passing the material when generating pipeline args.
      const pipeline = getPipelineForPrimitive({
        topology: TinyGltfWebGpu.gpuPrimitiveTopologyForMode(primitive.mode),
        buffers: sortedBufferLayout,
        doubleSided: material.doubleSided,
        alphaMode: material.alphaMode,
        // These values specifically will be passed to shader module creation.
        shaderArgs: {
          hasTexcoord: 'TEXCOORD_0' in primitive.attributes,
          useAlphaCutoff: material.alphaMode == 'MASK',
        },
      });

      // Rather than just storing a list of primitives for each pipeline store a map of
      // materials which use the pipeline to the primitives that use the material.
      let materialPrimitives = pipeline.materialPrimitives.get(gpuMaterial);
      if (!materialPrimitives) {
        materialPrimitives = [];
        pipeline.materialPrimitives.set(gpuMaterial, materialPrimitives);
      }

      // Add the primitive to the list of primitives for this material.
      materialPrimitives.push(gpuPrimitive);
    }
  }

  instanceBuffer.unmap();

  const instanceBindGroup = device.createBindGroup({
    label: `glTF Instance BindGroup`,
    layout: instanceBindGroupLayout,
    entries: [{
      binding: 0, // Instance storage buffer
      resource: {buffer: instanceBuffer},
    }],
  });

  function getShaderModule(args) {
    const key = JSON.stringify(args);

    let shaderModule = shaderModules.get(key);
    if (!shaderModule) {
      const code = wgsl`
              struct Camera {
                projection : mat4x4f,
                view : mat4x4f,
                position : vec3f,
                time : f32,
              };
              @group(0) @binding(0) var<uniform> camera : Camera;

              struct Model {
                matrix: mat4x4f,
                normalMat: mat4x4f,
              }
              @group(1) @binding(0) var<storage> instances : array<Model>;

              struct Material {
                baseColorFactor : vec4f,
                alphaCutoff: f32,
              };
              @group(2) @binding(0) var<uniform> material : Material;
              @group(2) @binding(1) var materialSampler : sampler;
              @group(2) @binding(2) var baseColorTexture : texture_2d<f32>;

              struct VertexInput {
                @builtin(instance_index) instance : u32,
                @location(${ShaderLocations.POSITION}) position : vec3f,
                @location(${ShaderLocations.NORMAL}) normal : vec3f,

                #if ${args.hasTexcoord}
                  @location(${ShaderLocations.TEXCOORD_0}) texcoord : vec2f,
                #endif
              };

              struct VertexOutput {
                @builtin(position) position : vec4f,
                @location(0) normal : vec3f,
                @location(1) texcoord : vec2f,
              };

              @vertex
              fn vertexMain(input : VertexInput) -> VertexOutput {
                var output : VertexOutput;

                let model = instances[input.instance];
                output.position = camera.projection * camera.view * model.matrix * vec4f(input.position, 1);
                output.normal = (camera.view * model.normalMat * vec4f(input.normal, 0)).xyz;

                #if ${args.hasTexcoord}
                  output.texcoord = input.texcoord;
                #else
                  output.texcoord = vec2f(0);
                #endif

                return output;
              }

              // Some hardcoded lighting
              const lightDir = vec3f(0.25, 0.5, 1);
              const lightColor = vec3f(1);
              const ambientColor = vec3f(0.1);

              @fragment
              fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {
                let baseColor = textureSample(baseColorTexture, materialSampler, input.texcoord) * material.baseColorFactor;

                #if ${args.useAlphaCutoff}
                  // If the alpha mode is MASK discard any fragments below the alpha cutoff.
                  if (baseColor.a < material.alphaCutoff) {
                    discard;
                  }
                #endif

                // An extremely simple directional lighting model, just to give our model some shape.
                let N = normalize(input.normal);
                let L = normalize(lightDir);
                let NDotL = max(dot(N, L), 0.0);
                let surfaceColor = (baseColor.rgb * ambientColor) + (baseColor.rgb * NDotL);

                return vec4f(surfaceColor, baseColor.a);
              }
            `;

      shaderModule = device.createShaderModule({
        label: 'Simple glTF rendering shader module',
        code,
      });
      shaderModules.set(key, shaderModule);
    }

    return shaderModule;
  }


  function getPipelineForPrimitive(args) {
    const key = JSON.stringify(args);

    const pipeline = pipelineGpuData.get(key);
    if (pipeline) {
      return pipeline;
    }

    const module = getShaderModule(args.shaderArgs);

    const gpuPipeline = {
      pipeline: device.createRenderPipeline({
        label: 'glTF renderer pipeline',
        layout: device.createPipelineLayout({
          label: 'glTF Pipeline Layout',
          bindGroupLayouts: [
            frameBindGroupLayout,
            instanceBindGroupLayout,
            materialBindGroupLayout,
          ]
        }),
        vertex: {
          module,
          entryPoint: 'vertexMain',
          buffers: args.buffers,
        },
        primitive: {
          topology: args.topology,
          // Make sure to apply the appropriate culling mode
          cullMode: args.doubleSided ? 'none' : 'back',
        },
        multisample: {
          count: 1,
        },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less',
        },
        fragment: {
          module,
          entryPoint: 'fragmentMain',
          targets: [{
            format: colorFormat,
            // Apply the necessary blending
            blend: args.alphaMode === 'BLEND' ? {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
              },
              alpha: {
                // This just prevents the canvas from having alpha "holes" in it.
                srcFactor: 'one',
                dstFactor: 'one',
              }
            } : undefined
          }],
        },
      }),
      // Cache a map of materials to the primitives that used them for each pipeline.
      materialPrimitives: new Map(),
    };

    pipelineGpuData.set(key, gpuPipeline);
    return gpuPipeline;
  }

  const size = {width: canvas.width, height: canvas.height};
  const colorAttachment = {
    // Appropriate target will be populated in onFrame
    view: undefined,
    resolveTarget: undefined,

    clearValue: {r: 0.0, g: 0.0, b: 0.2, a: 1.0},
    loadOp: 'clear',
    storeOp: 'store',
  };

  const depthStencilAttachment = {
    view: device.createTexture({
      size,
      sampleCount: 1,
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    }).createView(),
    depthClearValue: 1.0,
    depthLoadOp: 'clear',
    depthStoreOp: 'discard',
  };

  function frameCallback() {
    requestAnimationFrame(frameCallback);

    device.queue.writeBuffer(frameUniformBuffer, 0, frameArrayBuffer);

    const aspect = canvas.width / canvas.height;
    // Using mat4.perspectiveZO instead of mat4.perpective because WebGPU's
    // normalized device coordinates Z range is [0, 1], instead of WebGL's [-1, 1]
    mat4.perspectiveZO(projectionMatrix, Math.PI * 0.5, aspect, 0.01, sceneAabb.radius * 4.0);
    const commandEncoder = device.createCommandEncoder();
    colorAttachment.view = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [colorAttachment],
      depthStencilAttachment: depthStencilAttachment
    });

    renderPass.setBindGroup(0, frameBindGroup);
    renderPass.setBindGroup(1, instanceBindGroup);

    for (const gpuPipeline1 of pipelineGpuData.values()) {
      renderPass.setPipeline(gpuPipeline1.pipeline);

      // Loop through every material that uses this pipeline and get an array of primitives
      // that uses that material.
      for (const [material, primitives] of gpuPipeline1.materialPrimitives.entries()) {
        // Set the material bind group.
        renderPass.setBindGroup(2, material.bindGroup);

        // Loop through the primitives that use the current material/pipeline combo and draw
        // them as usual.
        for (const gpuPrimitive of primitives) {
          for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
            renderPass.setVertexBuffer(Number.parseInt(bufferIndex), gpuBuffer.buffer, gpuBuffer.offset);
          }
          if (gpuPrimitive.indexBuffer) {
            renderPass.setIndexBuffer(gpuPrimitive.indexBuffer, gpuPrimitive.indexType, gpuPrimitive.indexOffset);
          }

          if (gpuPrimitive.indexBuffer) {
            renderPass.drawIndexed(gpuPrimitive.drawCount, gpuPrimitive.instances.count, 0, 0, gpuPrimitive.instances.first);
          } else {
            renderPass.draw(gpuPrimitive.drawCount, gpuPrimitive.instances.count, 0, gpuPrimitive.instances.first);
          }
        }
      }
    }

    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
  }

  // Start the render loop.
  requestAnimationFrame(frameCallback);
}

function orbitCamera(element, target, maxDistance, minDistance, distance, callback) {
  let orbitX = 0;
  let orbitY = 0;

  function broadcast() {
    const mv = mat4.create();
    mat4.translate(mv, mv, target);
    mat4.rotateY(mv, mv, -orbitY);
    mat4.rotateX(mv, mv, -orbitX);
    mat4.translate(mv, mv, vec3.fromValues(0, 0, distance));
    mat4.invert(mv, mv);
    callback(mv)
  }

  broadcast();

  function pointerMove(event) {
    if (event.movementX || event.movementY) {
      orbitY += event.movementX * 0.025;
      while (orbitY < -Math.PI) {
        orbitY += Math.PI * 2;
      }
      while (orbitY >= Math.PI) {
        orbitY -= Math.PI * 2;
      }

      orbitX = Math.min(Math.max(orbitX + event.movementY * 0.025, -Math.PI * 0.5), Math.PI * 0.5);
      broadcast();
    }
  }

  function pointerUp(event) {
    element.removeEventListener('pointermove', pointerMove);
    element.removeEventListener('pointerup', pointerUp);
  }

  element.addEventListener('pointerdown', event => {
    element.addEventListener('pointermove', pointerMove);
    element.addEventListener('pointerup', pointerUp);
  });

  element.addEventListener('mousewheel', event => {
    const wheelDeltaY = event.wheelDeltaY;
    if (wheelDeltaY) {
      distance =
          Math.min(Math.max(distance - wheelDeltaY * 0.005, minDistance), maxDistance);
      broadcast();
    }
    event.preventDefault();
  });

}
