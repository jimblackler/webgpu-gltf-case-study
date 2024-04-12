import {mat4, vec3} from 'https://cdn.jsdelivr.net/npm/gl-matrix@3.4.3/esm/index.js';
import {TinyGltfWebGpu} from './tiny-gltf.js'
import {QueryArgs} from './query-args.js'
import {wgsl} from 'https://cdn.jsdelivr.net/npm/wgsl-preprocessor@1.0/wgsl-preprocessor.js'

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

function createSolidColorTexture(device, r, g, b, a) {
  const data = new Uint8Array([r * 255, g * 255, b * 255, a * 255]);
  const texture = device.createTexture({
    size: {width: 1, height: 1},
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });
  device.queue.writeTexture({texture}, data, {}, {width: 1, height: 1});
  return texture;
}

class GltfRenderer {
  pipelineGpuData = new Map();
  shaderModules = new Map();

  constructor(demoApp, gltf) {
    this.app = demoApp;
    this.device = demoApp.device;

    this.instanceBindGroupLayout = this.device.createBindGroupLayout({
      label: `glTF Instance BindGroupLayout`,
      entries: [{
        binding: 0, // Node uniforms
        visibility: GPUShaderStage.VERTEX,
        buffer: {type: 'read-only-storage'},
      }],
    });

    this.materialBindGroupLayout = this.device.createBindGroupLayout({
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

    this.gltfPipelineLayout = this.device.createPipelineLayout({
      label: 'glTF Pipeline Layout',
      bindGroupLayouts: [
        this.app.frameBindGroupLayout,
        this.instanceBindGroupLayout,
        this.materialBindGroupLayout,
      ]
    });

    const primitiveInstances = {
      matrices: new Map(),
      total: 0,
      arrayBuffer: null,
      offset: 0,
    };

    for (const node of gltf.nodes) {
      if ('mesh' in node) {
        this.setupMeshNode(gltf, node, primitiveInstances);
      }
    }

    this.opaqueWhiteTexture = createSolidColorTexture(this.device, 1, 1, 1, 1);

    const materialGpuData = new Map();
    for (const material of gltf.materials) {
      this.setupMaterial(gltf, material, materialGpuData);
    }

    // Create a buffer large enough to contain all the instance matrices for the entire scene.
    const instanceBuffer = this.device.createBuffer({
      size: 32 * Float32Array.BYTES_PER_ELEMENT * primitiveInstances.total,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    primitiveInstances.arrayBuffer = new Float32Array(instanceBuffer.getMappedRange());

    for (const mesh of gltf.meshes) {
      for (const primitive of mesh.primitives) {
        this.setupPrimitive(gltf, primitive, primitiveInstances, materialGpuData);
      }
    }

    instanceBuffer.unmap();

    this.instanceBindGroup = this.device.createBindGroup({
      label: `glTF Instance BindGroup`,
      layout: this.instanceBindGroupLayout,
      entries: [{
        binding: 0, // Instance storage buffer
        resource: {buffer: instanceBuffer},
      }],
    });
  }

  getShaderModule(args) {
    const key = JSON.stringify(args);

    let shaderModule = this.shaderModules.get(key);
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

      shaderModule = this.device.createShaderModule({
        label: 'Simple glTF rendering shader module',
        code,
      });
      this.shaderModules.set(key, shaderModule);
    }

    return shaderModule;
  }

  setupMaterial(gltf, material, materialGpuData) {
    // Create a uniform buffer for this material and populate it with the material properties.
    const materialUniformBuffer = this.device.createBuffer({
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
      baseColor = {
        texture: this.opaqueWhiteTexture,
        sampler: gltf.gpuDefaultSampler,
      };
    }

    const bindGroup = this.device.createBindGroup({
      label: `glTF Material BindGroup`,
      layout: this.materialBindGroupLayout,
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
    });

    materialGpuData.set(material, {
      bindGroup,
    });
  }

  setupMeshNode(gltf, node, primitiveInstances) {
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

  setupPrimitiveInstances(primitive, primitiveInstances) {
    const instances = primitiveInstances.matrices.get(primitive);

    const first = primitiveInstances.offset;
    const count = instances.length;

    for (let i = 0; i < count; ++i) {
      primitiveInstances.arrayBuffer.set(instances[i].worldMatrix, (first + i) * 32);
      primitiveInstances.arrayBuffer.set(instances[i].normalMatrix, (first + i) * 32 + 16);
    }

    primitiveInstances.offset += count;

    return {first, count};
  }

  setupPrimitive(gltf, primitive, primitiveInstances, materialGpuData) {
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
      buffer.attributes = buffer.attributes.sort((a, b) => {
        return a.shaderLocation - b.shaderLocation;
      });
    }
    // Sort the buffers by their first attribute's shader location.
    const sortedBufferLayout = [...bufferLayout.values()].sort((a, b) => {
      return a.attributes[0].shaderLocation - b.attributes[0].shaderLocation;
    });

    // Ensure that the gpuBuffers are saved in the same order as the buffer layout.
    const sortedGpuBuffers = [];
    for (const buffer of sortedBufferLayout) {
      sortedGpuBuffers.push(gpuBuffers.get(buffer));
    }

    const gpuPrimitive = {
      buffers: sortedGpuBuffers,
      drawCount,
      instances: this.setupPrimitiveInstances(primitive, primitiveInstances),
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
    const pipelineArgs = this.getPipelineArgs(primitive, sortedBufferLayout, material);
    const pipeline = this.getPipelineForPrimitive(pipelineArgs);

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

  getPipelineArgs(primitive, buffers, material) {
    return {
      topology: TinyGltfWebGpu.gpuPrimitiveTopologyForMode(primitive.mode),
      buffers,
      doubleSided: material.doubleSided,
      alphaMode: material.alphaMode,
      // These values specifically will be passed to shader module creation.
      shaderArgs: {
        hasTexcoord: 'TEXCOORD_0' in primitive.attributes,
        useAlphaCutoff: material.alphaMode == 'MASK',
      },
    };
  }

  getPipelineForPrimitive(args) {
    const key = JSON.stringify(args);

    let pipeline = this.pipelineGpuData.get(key);
    if (pipeline) {
      return pipeline;
    }

    // Define the alpha blending behavior.
    let blend = undefined;
    if (args.alphaMode == 'BLEND') {
      blend = {
        color: {
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
        },
        alpha: {
          // This just prevents the canvas from having alpha "holes" in it.
          srcFactor: 'one',
          dstFactor: 'one',
        }
      }
    }

    const module = this.getShaderModule(args.shaderArgs);
    pipeline = this.device.createRenderPipeline({
      label: 'glTF renderer pipeline',
      layout: this.gltfPipelineLayout,
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
        format: this.app.depthFormat,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      fragment: {
        module,
        entryPoint: 'fragmentMain',
        targets: [{
          format: this.app.colorFormat,
          // Apply the necessary blending
          blend,
        }],
      },
    });

    const gpuPipeline = {
      pipeline,
      // Cache a map of materials to the primitives that used them for each pipeline.
      materialPrimitives: new Map(),
    };

    this.pipelineGpuData.set(key, gpuPipeline);

    return gpuPipeline;
  }

  render(renderPass) {
    renderPass.setBindGroup(0, this.app.frameBindGroup);
    renderPass.setBindGroup(1, this.instanceBindGroup);

    for (const gpuPipeline of this.pipelineGpuData.values()) {
      renderPass.setPipeline(gpuPipeline.pipeline);

      // Loop through every material that uses this pipeline and get an array of primitives
      // that uses that material.
      for (const [material, primitives] of gpuPipeline.materialPrimitives.entries()) {
        // Set the material bind group.
        renderPass.setBindGroup(2, material.bindGroup);

        // Loop through the primitives that use the current material/pipeline combo and draw
        // them as usual.
        for (const gpuPrimitive of primitives) {
          for (const [bufferIndex, gpuBuffer] of Object.entries(gpuPrimitive.buffers)) {
            renderPass.setVertexBuffer(bufferIndex, gpuBuffer.buffer, gpuBuffer.offset);
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
  }
}


// Style for elements used by the demo.
const injectedStyle = document.createElement('style');
injectedStyle.innerText = `
  canvas {
    position: absolute;
    z-index: 0;
    height: 100%;
    width: 100%;
    inset: 0;
    margin: 0;
    touch-action: none;
  }`;
document.head.appendChild(injectedStyle);

const FRAME_BUFFER_SIZE = Float32Array.BYTES_PER_ELEMENT * 36;

export class GltfDemo {
  clearColor = {r: 0.0, g: 0.0, b: 0.2, a: 1.0};

  rendererClass = null;
  gltfRenderer = null;

  colorFormat = navigator.gpu?.getPreferredCanvasFormat?.() || 'bgra8unorm';
  depthFormat = 'depth24plus';
  #frameArrayBuffer = new ArrayBuffer(FRAME_BUFFER_SIZE);
  #projectionMatrix = new Float32Array(this.#frameArrayBuffer, 0, 16);
  #viewMatrix = new Float32Array(this.#frameArrayBuffer, 16 * Float32Array.BYTES_PER_ELEMENT, 16);
  #cameraPosition = new Float32Array(this.#frameArrayBuffer, 32 * Float32Array.BYTES_PER_ELEMENT, 3);
  #timeArray = new Float32Array(this.#frameArrayBuffer, 35 * Float32Array.BYTES_PER_ELEMENT, 1);

  fov = Math.PI * 0.5;
  zNear = 0.01;
  zFar = 128;

  #frameMs = new Array(20);
  #frameMsIndex = 0;


  constructor(startup_model) {

    this.canvas = document.querySelector('.webgpu-canvas');

    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      document.body.appendChild(this.canvas);
    }
    this.context = this.canvas.getContext('webgpu');

    this.camera = new OrbitCamera(this.canvas);

    this.resizeObserver = new ResizeObserverHelper(this.canvas, (width, height) => {
      if (width == 0 || height == 0) {
        return;
      }

      this.canvas.width = width;
      this.canvas.height = height;

      this.updateProjection();

      if (this.device) {
        const size = {width, height};
        this.#allocateRenderTargets(size);
        this.onResize(this.device, size);
      }
    });

    const frameCallback = (t) => {
      requestAnimationFrame(frameCallback);

      const frameStart = performance.now();

      // Update the frame uniforms
      this.#viewMatrix.set(this.camera.viewMatrix);
      this.#cameraPosition.set(this.camera.position);
      this.#timeArray[0] = t;

      this.device.queue.writeBuffer(this.frameUniformBuffer, 0, this.#frameArrayBuffer);

      this.onFrame(this.device, this.context, t);

      this.#frameMs[this.#frameMsIndex++ % this.#frameMs.length] = performance.now() - frameStart;
    };

    this.#initWebGPU().then(() => {
      // Make sure the resize callback has a chance to fire at least once now that the device is
      // initialized.
      this.resizeObserver.callback(this.canvas.width, this.canvas.height);
      // Start the render loop.
      requestAnimationFrame(frameCallback);
    }).catch((error) => {
      // If something goes wrong during initialization, put up a really simple error message.
      this.setError(error, 'initializing WebGPU');
      throw error;
    });
    // Allow the startup model to be overriden by a query arg.
    startup_model = QueryArgs.getString('model', startup_model);
    if (startup_model in GltfModels) {
      this.model = GltfModels[startup_model];
    } else {
      this.model = GltfModels['antique_camera'];
    }

    this.rendererClass = rendererClass;

  }

  onInit(device) {
    this.gltfLoader = new TinyGltfWebGpu(device);

    this.onLoadModel(device, this.model);
  }

  async onLoadModel(device, url) {
    console.log('Loading', url);

    const gltf = await this.gltfLoader.loadFromUrl(url);
    const sceneAabb = gltf.scenes[gltf.scene].aabb;

    this.camera.target = sceneAabb.center;
    this.camera.maxDistance = sceneAabb.radius * 2.0;
    this.camera.minDistance = sceneAabb.radius * 0.25;
    if (url.includes('Sponza')) {
      this.camera.distance = this.camera.minDistance;
    } else {
      this.camera.distance = sceneAabb.radius * 1.5;
    }
    this.zFar = sceneAabb.radius * 4.0;

    this.updateProjection();

    console.log(gltf);

    try {
      device.pushErrorScope('validation');

      this.gltfRenderer = new GltfRenderer(this, gltf);

      device.popErrorScope().then((error) => {
        this.setError(error, 'loading glTF model');
        if (error) {
          this.gltfRenderer = null;
        }
      });
    } catch (error) {
      this.setError(error, 'loading glTF model');
      throw error;
    }
  }

  onFrame(device, context, timestamp) {
    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass(this.defaultRenderPassDescriptor);

    if (this.gltfRenderer) {
      this.gltfRenderer.render(renderPass);
    }

    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
  }

  setError(error, contextString) {
    let prevError = document.querySelector('.error');
    while (prevError) {
      this.canvas.parentElement.removeChild(document.querySelector('.error'));
      prevError = document.querySelector('.error');
    }

    if (error) {
      const errorElement = document.createElement('p');
      errorElement.classList.add('error');
      errorElement.innerHTML = `
        <p style='font-weight: bold'>An error occured${contextString ? ' while ' + contextString : ''}:</p>
        <pre>${error?.message ? error.message : error}</pre>`;
      this.canvas.parentElement.appendChild(errorElement);
    }
  }

  updateProjection() {
    const aspect = this.canvas.width / this.canvas.height;
    // Using mat4.perspectiveZO instead of mat4.perpective because WebGPU's
    // normalized device coordinates Z range is [0, 1], instead of WebGL's [-1, 1]
    mat4.perspectiveZO(this.#projectionMatrix, this.fov, aspect, this.zNear, this.zFar);
  }

  get frameMs() {
    let avg = 0;
    for (const value of this.#frameMs) {
      if (value === undefined) {
        return 0;
      } // Don't have enough sampled yet
      avg += value;
    }
    return avg / this.#frameMs.length;
  }

  async #initWebGPU() {
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();
    this.context.configure({
      device: this.device,
      format: this.colorFormat,
      alphaMode: 'opaque',
    });

    this.frameUniformBuffer = this.device.createBuffer({
      size: FRAME_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.frameBindGroupLayout = this.device.createBindGroupLayout({
      label: `Frame BindGroupLayout`,
      entries: [{
        binding: 0, // Camera/Frame uniforms
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {},
      }],
    });

    this.frameBindGroup = this.device.createBindGroup({
      label: `Frame BindGroup`,
      layout: this.frameBindGroupLayout,
      entries: [{
        binding: 0, // Camera uniforms
        resource: {buffer: this.frameUniformBuffer},
      }],
    });

    await this.onInit(this.device);
  }

  #allocateRenderTargets(size) {
    if (this.msaaColorTexture) {
      this.msaaColorTexture.destroy();
    }

    if (this.depthTexture) {
      this.depthTexture.destroy();
    }

    this.depthTexture = this.device.createTexture({
      size,
      sampleCount: 1,
      format: this.depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.colorAttachment = {
      // Appropriate target will be populated in onFrame
      view: undefined,
      resolveTarget: undefined,

      clearValue: this.clearColor,
      loadOp: 'clear',
      storeOp: 'store',
    };

    this.renderPassDescriptor = {
      colorAttachments: [this.colorAttachment],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      }
    };
  }

  get defaultRenderPassDescriptor() {
    this.colorAttachment.view = this.context.getCurrentTexture().createView();
    return this.renderPassDescriptor;
  }


  onResize(device, size) {
    // Override to handle resizing logic
  }

}

class ResizeObserverHelper extends ResizeObserver {
  constructor(element, callback) {
    super(entries => {
      for (let entry of entries) {
        if (entry.target != element) {
          continue;
        }

        if (entry.devicePixelContentBoxSize) {
          // Should give exact pixel dimensions, but only works on Chrome.
          const devicePixelSize = entry.devicePixelContentBoxSize[0];
          callback(devicePixelSize.inlineSize, devicePixelSize.blockSize);
        } else if (entry.contentBoxSize) {
          // Firefox implements `contentBoxSize` as a single content rect, rather than an array
          const contentBoxSize = Array.isArray(entry.contentBoxSize) ? entry.contentBoxSize[0] : entry.contentBoxSize;
          callback(contentBoxSize.inlineSize, contentBoxSize.blockSize);
        } else {
          callback(entry.contentRect.width, entry.contentRect.height);
        }
      }
    });

    this.element = element;
    this.callback = callback;

    this.observe(element);
  }
}

export class OrbitCamera {
  orbitX = 0;
  orbitY = 0;
  maxOrbitX = Math.PI * 0.5;
  minOrbitX = -Math.PI * 0.5;
  maxOrbitY = Math.PI;
  minOrbitY = -Math.PI;
  constrainXOrbit = true;
  constrainYOrbit = false;

  maxDistance = 10;
  minDistance = 1;
  distanceStep = 0.005;
  constrainDistance = true;

  #distance = vec3.create([0, 0, 5]);
  #target = vec3.create();
  #viewMat = mat4.create();
  #cameraMat = mat4.create();
  #position = vec3.create();
  #dirty = true;

  #element;
  #registerElement;

  constructor(element = null) {
    let moving = false;
    let lastX, lastY;

    const downCallback = (event) => {
      if (event.isPrimary) {
        moving = true;
      }
      lastX = event.pageX;
      lastY = event.pageY;
    };
    const moveCallback = (event) => {
      let xDelta, yDelta;

      if (document.pointerLockEnabled) {
        xDelta = event.movementX;
        yDelta = event.movementY;
        this.orbit(xDelta * 0.025, yDelta * 0.025);
      } else if (moving) {
        xDelta = event.pageX - lastX;
        yDelta = event.pageY - lastY;
        lastX = event.pageX;
        lastY = event.pageY;
        this.orbit(xDelta * 0.025, yDelta * 0.025);
      }
    };
    const upCallback = (event) => {
      if (event.isPrimary) {
        moving = false;
      }
    };
    const wheelCallback = (event) => {
      this.distance = this.#distance[2] + (-event.wheelDeltaY * this.distanceStep);
      event.preventDefault();
    };

    this.#registerElement = (value) => {
      if (this.#element && this.#element != value) {
        this.#element.removeEventListener('pointerdown', downCallback);
        this.#element.removeEventListener('pointermove', moveCallback);
        this.#element.removeEventListener('pointerup', upCallback);
        this.#element.removeEventListener('mousewheel', wheelCallback);
      }

      this.#element = value;
      if (this.#element) {
        this.#element.addEventListener('pointerdown', downCallback);
        this.#element.addEventListener('pointermove', moveCallback);
        this.#element.addEventListener('pointerup', upCallback);
        this.#element.addEventListener('mousewheel', wheelCallback);
      }
    }

    this.#element = element;
    this.#registerElement(element);
  }

  set element(value) {
    this.#registerElement(value);
  }

  get element() {
    return this.#element;
  }

  orbit(xDelta, yDelta) {
    if (xDelta || yDelta) {
      this.orbitY += xDelta;
      if (this.constrainYOrbit) {
        this.orbitY = Math.min(Math.max(this.orbitY, this.minOrbitY), this.maxOrbitY);
      } else {
        while (this.orbitY < -Math.PI) {
          this.orbitY += Math.PI * 2;
        }
        while (this.orbitY >= Math.PI) {
          this.orbitY -= Math.PI * 2;
        }
      }

      this.orbitX += yDelta;
      if (this.constrainXOrbit) {
        this.orbitX = Math.min(Math.max(this.orbitX, this.minOrbitX), this.maxOrbitX);
      } else {
        while (this.orbitX < -Math.PI) {
          this.orbitX += Math.PI * 2;
        }
        while (this.orbitX >= Math.PI) {
          this.orbitX -= Math.PI * 2;
        }
      }

      this.#dirty = true;
    }
  }

  get target() {
    return [this.#target[0], this.#target[1], this.#target[2]];
  }

  set target(value) {
    this.#target[0] = value[0];
    this.#target[1] = value[1];
    this.#target[2] = value[2];
    this.#dirty = true;
  };

  get distance() {
    return -this.#distance[2];
  };

  set distance(value) {
    this.#distance[2] = value;
    if (this.constrainDistance) {
      this.#distance[2] = Math.min(Math.max(this.#distance[2], this.minDistance), this.maxDistance);
    }
    this.#dirty = true;
  };

  #updateMatrices() {
    if (this.#dirty) {
      var mv = this.#cameraMat;
      mat4.identity(mv);

      mat4.translate(mv, mv, this.#target);
      mat4.rotateY(mv, mv, -this.orbitY);
      mat4.rotateX(mv, mv, -this.orbitX);
      mat4.translate(mv, mv, this.#distance);
      mat4.invert(this.#viewMat, this.#cameraMat);

      this.#dirty = false;
    }
  }

  get position() {
    this.#updateMatrices();
    vec3.set(this.#position, 0, 0, 0);
    vec3.transformMat4(this.#position, this.#position, this.#cameraMat);
    return this.#position;
  }

  get viewMatrix() {
    this.#updateMatrices();
    return this.#viewMat;
  }
}
