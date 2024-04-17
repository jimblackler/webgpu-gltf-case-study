import {mat4, vec3} from "../node_modules/gl-matrix/esm/index.js";
import {wgsl} from "../node_modules/wgsl-preprocessor/wgsl-preprocessor.js"

// To make it easier to reference the WebGL enums that glTF uses.
const GL = WebGLRenderingContext;

const GltfRootDir = "./glTF-Sample-Models/2.0";

const GltfModels = {
  antique_camera: `${GltfRootDir}/AntiqueCamera/glTF/AntiqueCamera.gltf`,
  buggy: `${GltfRootDir}/Buggy/glTF-Binary/Buggy.glb`,
  corset: `${GltfRootDir}/Corset/glTF-Binary/Corset.glb`,
  damaged_helmet: `${GltfRootDir}/DamagedHelmet/glTF-Binary/DamagedHelmet.glb`,
  flight_helmet: `${GltfRootDir}/FlightHelmet/glTF/FlightHelmet.gltf`,
  sponza: `./sponza-optimized/Sponza.gltf`,
  mc_laren: `${GltfRootDir}/McLaren.glb`,
  porsche_gt3_rs: `${GltfRootDir}/porsche_gt3_rs.glb`,
  sea_keep_lonely_watcher: `${GltfRootDir}/sea_keep_lonely_watcher.glb`,
  shiba: `${GltfRootDir}/shiba.glb`
}

// Shader locations and source are unchanged from the previous sample.
const ShaderLocations = {
  POSITION: 0,
  NORMAL: 1,
  // Add texture coordinates to the list of attributes we care about.
  TEXCOORD_0: 2
};

const GLB_MAGIC = 0x46546C67;
const JSON_CHUNK_TYPE = 0x4E4F534A;
const BIN_CHUNK_TYPE = 0x004E4942;

const FRAME_BUFFER_SIZE = Float32Array.BYTES_PER_ELEMENT * 36;

function componentCountForType(type) {
  switch (type) {
    case "SCALAR":
      return 1;
    case "VEC2":
      return 2;
    case "VEC3":
      return 3;
    case "VEC4":
      return 4;
    default:
      return 0;
  }
}

function sizeForComponentType(componentType) {
  switch (componentType) {
    case GL.BYTE:
      return 1;
    case GL.UNSIGNED_BYTE:
      return 1;
    case GL.SHORT:
      return 2;
    case GL.UNSIGNED_SHORT:
      return 2;
    case GL.UNSIGNED_INT:
      return 4;
    case GL.FLOAT:
      return 4;
    default:
      return 0;
  }
}

function packedArrayStrideForAccessor(accessor) {
  return sizeForComponentType(accessor.componentType) * componentCountForType(accessor.type);
}

function gpuFormatForAccessor(accessor) {
  const norm = accessor.normalized ? "norm" : "int";
  const count = componentCountForType(accessor.type);
  const x = count > 1 ? `x${count}` : "";
  switch (accessor.componentType) {
    case GL.BYTE:
      return `s${norm}8${x}`;
    case GL.UNSIGNED_BYTE:
      return `u${norm}8${x}`;
    case GL.SHORT:
      return `s${norm}16${x}`;
    case GL.UNSIGNED_SHORT:
      return `u${norm}16${x}`;
    case GL.UNSIGNED_INT:
      return `u${norm}32${x}`;
    case GL.FLOAT:
      return `float32${x}`;
  }
}

function gpuPrimitiveTopologyForMode(mode) {
  switch (mode) {
    case GL.TRIANGLES:
      return "triangle-list";
    case GL.TRIANGLE_STRIP:
      return "triangle-strip";
    case GL.LINES:
      return "line-list";
    case GL.LINE_STRIP:
      return "line-strip";
    case GL.POINTS:
      return "point-list";
  }
}

function gpuIndexFormatForComponentType(componentType) {
  switch (componentType) {
    case GL.UNSIGNED_SHORT:
      return "uint16";
    case GL.UNSIGNED_INT:
      return "uint32";
    default:
      return 0;
  }
}

function getNormalMatrixMap(worldMatrixMap) {
  return new Map(worldMatrixMap.entries().map(([node, worldMatrix]) => {
        const normalMatrix = mat4.clone(worldMatrix);
        normalMatrix[12] = 0;
        normalMatrix[13] = 0;
        normalMatrix[14] = 0;
        mat4.transpose(normalMatrix, mat4.invert(normalMatrix, normalMatrix));
        return [node, normalMatrix];
      }
  ));
}

function getWorldMatrixMap(gltf) {
  const worldMatrixMap = new Map();

  function setWorldMatrix(nodeIdxs, parentWorldMatrix) {
    for (const node of nodeIdxs.map(nodeIdx => gltf.nodes[nodeIdx])) {
      // Don't recompute nodes we've already visited.
      if (worldMatrixMap.has(node)) {
        continue;
      }

      let worldMatrix;
      if (node.matrix) {
        worldMatrix = mat4.clone(node.matrix);
      } else {
        worldMatrix = mat4.create();
        if (node.rotation || node.position || node.translation) {
          mat4.fromRotationTranslationScale(
              worldMatrix,
              node.rotation ?? vec3.fromValues(0, 0, 0),
              node.translation ?? vec3.fromValues(0, 0, 0),
              node.scale ?? vec3.fromValues(1, 1, 1));
        }
      }

      mat4.multiply(worldMatrix, parentWorldMatrix, worldMatrix);
      worldMatrixMap.set(node, worldMatrix);

      setWorldMatrix(node.children ?? [], worldMatrix);
    }
  }

  // Compute a world transform for each node, starting at the root nodes and
  // working our way down.
  for (const scene of Object.values(gltf.scenes)) {
    setWorldMatrix(scene.nodes, mat4.create());
  }
  return worldMatrixMap;
}

function gpuAddressModeForWrap(wrap) {
  switch (wrap) {
    case GL.CLAMP_TO_EDGE:
      return "clamp-to-edge";
    case GL.MIRRORED_REPEAT:
      return "mirror-repeat";
    default:
      return "repeat";
  }
}

function createGpuBufferFromBufferView(device, bufferView, buffer, usage) {
  // For our purposes we're only worried about bufferViews that have a vertex or index usage.
  if (!usage) {
    return null;
  }

  const gpuBuffer = device.createBuffer({
    label: bufferView.name,
    // Round the buffer size up to the nearest multiple of 4.
    size: Math.ceil(bufferView.byteLength / 4) * 4,
    usage: usage,
    mappedAtCreation: true,
  });

  const gpuBufferArray = new Uint8Array(gpuBuffer.getMappedRange());
  gpuBufferArray.set(new Uint8Array(buffer, bufferView.byteOffset, bufferView.byteLength));
  gpuBuffer.unmap();

  return gpuBuffer;
}

function createGpuSamplerFromSampler(device, sampler) {
  const descriptor = {
    label: sampler.name,
    addressModeU: gpuAddressModeForWrap(sampler.wrapS),
    addressModeV: gpuAddressModeForWrap(sampler.wrapT),
  };

  if (!sampler.magFilter || sampler.magFilter == GL.LINEAR) {
    descriptor.magFilter = "linear";
  }

  switch (sampler.minFilter) {
    case GL.NEAREST:
      break;
    case GL.LINEAR:
    case GL.LINEAR_MIPMAP_NEAREST:
      descriptor.minFilter = "linear";
      break;
    case GL.NEAREST_MIPMAP_LINEAR:
      descriptor.mipmapFilter = "linear";
      break;
    case GL.LINEAR_MIPMAP_LINEAR:
    default:
      descriptor.minFilter = "linear";
      descriptor.mipmapFilter = "linear";
      break;
  }

  return device.createSampler(descriptor);
}

function createGpuTextureFromImage(device, source) {
  const size = {width: source.width, height: source.height};
  const texture = device.createTexture({
    size: size,
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    mipLevelCount: 1
  });
  device.queue.copyExternalImageToTexture({source}, {texture}, size);

  return texture;
}


export async function gltfDemo(startup_model) {
  const colorFormat = navigator.gpu?.getPreferredCanvasFormat?.() || "bgra8unorm";
  const frameArrayBuffer = new ArrayBuffer(FRAME_BUFFER_SIZE);
  const projectionMatrix = new Float32Array(frameArrayBuffer, 0, 16);
  const viewMatrix = new Float32Array(frameArrayBuffer, 16 * Float32Array.BYTES_PER_ELEMENT, 16);
  const canvas = document.querySelector(".webgpu-canvas");

  const context = canvas.getContext("webgpu");

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  context.configure({
    device: device,
    format: colorFormat,
    alphaMode: "opaque",
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

  const url = GltfModels[startup_model];
  const response = await fetch(url);

  if (!url.endsWith(".glb")) {
    throw Error("Only loads glb")
  }
  const arrayBuffer = await response.arrayBuffer();

  const headerView = new DataView(arrayBuffer, 0, 12);
  const magic = headerView.getUint32(0, true);
  const version = headerView.getUint32(4, true);
  const length = headerView.getUint32(8, true);

  if (magic !== GLB_MAGIC) {
    throw new Error("Invalid magic string in binary header.");
  }

  if (version !== 2) {
    throw new Error("Incompatible version in binary header.");
  }

  let chunks = {};
  let chunkOffset = 12;
  while (chunkOffset < length) {
    const chunkHeaderView = new DataView(arrayBuffer, chunkOffset, 8);
    const chunkLength = chunkHeaderView.getUint32(0, true);
    const chunkType = chunkHeaderView.getUint32(4, true);
    chunks[chunkType] = arrayBuffer.slice(chunkOffset + 8, chunkOffset + 8 + chunkLength);
    chunkOffset += chunkLength + 8;
  }

  if (!chunks[JSON_CHUNK_TYPE]) {
    throw new Error("File contained no json chunk.");
  }

  const decoder = new TextDecoder("utf-8");
  const jsonString = decoder.decode(chunks[JSON_CHUNK_TYPE]);
  const gltf = JSON.parse(jsonString);
  if (!gltf.asset) {
    throw new Error("Missing asset description.");
  }

  if (gltf.asset.minVersion !== "2.0" && gltf.asset.version !== "2.0") {
    throw new Error("Incompatible asset version.");
  }

  // Resolve defaults for as many properties as we can.
  for (const accessor of gltf.accessors) {
    accessor.byteOffset = accessor.byteOffset ?? 0;
    accessor.normalized = accessor.normalized ?? false;
  }

  for (const bufferView of gltf.bufferViews) {
    bufferView.byteOffset = bufferView.byteOffset ?? 0;
  }

  for (const sampler of gltf.samplers ?? []) {
    sampler.wrapS = sampler.wrapS ?? GL.REPEAT;
    sampler.wrapT = sampler.wrapT ?? GL.REPEAT;
  }

  // Identify all the vertex and index buffers by iterating through all the primitives accessors
  // and marking the buffer views as vertex or index usage.
  // (There's technically a target attribute on the buffer view that's supposed to tell us what
  // it's used for, but that appears to be rarely populated.)
  const bufferViewUsages = [];

  function markAccessorUsage(accessorIndex, usage) {
    bufferViewUsages[gltf.accessors[accessorIndex].bufferView] |= usage;
  }

  for (const mesh of gltf.meshes) {
    for (const primitive of mesh.primitives) {
      if ("indices" in primitive) {
        markAccessorUsage(primitive.indices, GPUBufferUsage.INDEX);
      }
      for (const attribute of Object.values(primitive.attributes)) {
        markAccessorUsage(attribute, GPUBufferUsage.VERTEX);
      }
    }
  }

  const imageTextures = await Promise.all(gltf.images.map(image => {
    if (image.uri) {
      throw Error("Image URI fetching not supported");
    }
    const bufferView = gltf.bufferViews[image.bufferView];
    if (bufferView.buffer !== 0) {
      throw Error();
    }
    return createImageBitmap(new Blob(
        [new Uint8Array(
            chunks[BIN_CHUNK_TYPE], bufferView.byteOffset, bufferView.byteLength)],
        {type: image.mimeType})).then(image => createGpuTextureFromImage(device, image))
  }));

  const gpuSamplers = Object.values(gltf.samplers ?? []).map(sampler =>
      createGpuSamplerFromSampler(device, sampler));

  const gpuBuffers = Object.values(gltf.bufferViews).map((bufferView, index) =>
      createGpuBufferFromBufferView(device, bufferView,
          chunks[BIN_CHUNK_TYPE], bufferViewUsages[index]));
  const worldMatrixMap = getWorldMatrixMap(gltf);
  const normalMap = getNormalMatrixMap(worldMatrixMap);

  orbitCamera(canvas, vec3.fromValues(0, 0, 0), 1.5, mtx => viewMatrix.set(mtx));

  const pipelineGpuData = new Map();
  const shaderModules = new Map();

  const instanceBindGroupLayout = device.createBindGroupLayout({
    label: `glTF Instance BindGroupLayout`,
    entries: [{
      binding: 0, // Node uniforms
      visibility: GPUShaderStage.VERTEX,
      buffer: {type: "read-only-storage"},
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

  gltf.nodes.forEach(node => gltf.meshes[node.mesh]?.primitives.forEach(primitive => {
    const instances = primitiveInstances.matrices.get(primitive);
    if (instances) {
      instances.push(node);
    } else {
      primitiveInstances.matrices.set(primitive, [node]);
    }
    primitiveInstances.total++;
  }))

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

    const texture = gltf.textures[material.pbrMetallicRoughness?.baseColorTexture?.index];
    materialGpuData.set(material, {
      bindGroup: device.createBindGroup({
        label: `glTF Material BindGroup`,
        layout: materialBindGroupLayout,
        entries: [{
          binding: 0, // Material uniforms
          resource: {buffer: materialUniformBuffer},
        }, {
          binding: 1, // Sampler
          resource: gpuSamplers[texture.sampler],
        }, {
          binding: 2, // BaseColor
          resource: imageTextures[texture.source].createView()
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
      const _gpuBuffers = new Map();
      let drawCount = 0;

      for (const [attribName, accessorIndex] of Object.entries(primitive.attributes)) {
        const shaderLocation = ShaderLocations[attribName];
        if (shaderLocation === undefined) {
          continue;
        }

        const accessor = gltf.accessors[accessorIndex];
        let buffer = bufferLayout.get(accessor.bufferView);

        let separate = buffer &&
            Math.abs(accessor.byteOffset - buffer.attributes[0].offset) >= buffer.arrayStride;
        if (!buffer || separate) {
          buffer = {
            arrayStride: gltf.bufferViews[accessor.bufferView].byteStride ||
                packedArrayStrideForAccessor(accessor),
            attributes: [],
          };

          bufferLayout.set(separate ? attribName : accessor.bufferView, buffer);
          _gpuBuffers.set(buffer, {
            buffer: gpuBuffers[accessor.bufferView],
            offset: accessor.byteOffset
          });
        } else {
          const gpuBuffer = _gpuBuffers.get(buffer);
          gpuBuffer.offset = Math.min(gpuBuffer.offset, accessor.byteOffset);
        }

        buffer.attributes.push({
          shaderLocation,
          format: gpuFormatForAccessor(accessor),
          offset: accessor.byteOffset,
        });

        drawCount = accessor.count;
      }

      for (const buffer of bufferLayout.values()) {
        const gpuBuffer = _gpuBuffers.get(buffer);
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
      const sortedGpuBuffers = sortedBufferLayout.map(buffer => _gpuBuffers.get(buffer));

      const instances = primitiveInstances.matrices.get(primitive);

      instances.forEach((instance, i) => {
        const idx = primitiveInstances.offset + i;
        primitiveInstances.arrayBuffer.set(worldMatrixMap.get(instance), idx * 32);
        primitiveInstances.arrayBuffer.set(normalMap.get(instance), idx * 32 + 16);
      });

      primitiveInstances.offset += instances.length;
      const gpuPrimitive = {
        buffers: sortedGpuBuffers,
        drawCount,
        instances: primitiveInstances
      };

      if ("indices" in primitive) {
        const accessor = gltf.accessors[primitive.indices];
        gpuPrimitive.indexBuffer = gpuBuffers[accessor.bufferView];
        gpuPrimitive.indexOffset = accessor.byteOffset;
        gpuPrimitive.indexType = gpuIndexFormatForComponentType(accessor.componentType);
        gpuPrimitive.drawCount = accessor.count;
      }

      const material = gltf.materials[primitive.material];

      // Start passing the material when generating pipeline args.
      // Rather than just storing a list of primitives for each pipeline store a map of
      // materials which use the pipeline to the primitives that use the material.
      const materialPrimitives1 = getPipelineForPrimitive({
        topology: gpuPrimitiveTopologyForMode(primitive.mode),
        buffers: sortedBufferLayout,
        doubleSided: material.doubleSided,
        alphaMode: material.alphaMode,
        // These values specifically will be passed to shader module creation.
        shaderArgs: {
          hasTexcoord: "TEXCOORD_0" in primitive.attributes,
          useAlphaCutoff: material.alphaMode == "MASK",
        },
      }).materialPrimitives;

      // Add the primitive to the list of primitives for this material.
      const gpuMaterial = materialGpuData.get(material);
      const materialPrimitives = materialPrimitives1.get(gpuMaterial);
      if (materialPrimitives) {
        materialPrimitives.push(gpuPrimitive);
      } else {
        materialPrimitives1.set(gpuMaterial, [gpuPrimitive]);
      }
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
        label: "Simple glTF rendering shader module",
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
        label: "glTF renderer pipeline",
        layout: device.createPipelineLayout({
          label: "glTF Pipeline Layout",
          bindGroupLayouts: [
            frameBindGroupLayout,
            instanceBindGroupLayout,
            materialBindGroupLayout,
          ]
        }),
        vertex: {
          module,
          entryPoint: "vertexMain",
          buffers: args.buffers,
        },
        primitive: {
          topology: args.topology,
          // Make sure to apply the appropriate culling mode
          cullMode: args.doubleSided ? "none" : "back",
        },
        multisample: {
          count: 1,
        },
        depthStencil: {
          format: "depth24plus",
          depthWriteEnabled: true,
          depthCompare: "less",
        },
        fragment: {
          module,
          entryPoint: "fragmentMain",
          targets: [{
            format: colorFormat,
            // Apply the necessary blending
            blend: args.alphaMode === "BLEND" ? {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
              },
              alpha: {
                // This just prevents the canvas from having alpha "holes" in it.
                srcFactor: "one",
                dstFactor: "one",
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

  const colorAttachment = {
    // Appropriate target will be populated in onFrame
    view: undefined,

    clearValue: {r: 0.0, g: 0.0, b: 0.2, a: 1.0},
    loadOp: "clear",
    storeOp: "store",
  };

  const depthStencilAttachment = {
    view: device.createTexture({
      size: {width: canvas.width, height: canvas.height},
      sampleCount: 1,
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    }).createView(),
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "discard",
  };

  function frameCallback() {
    requestAnimationFrame(frameCallback);

    device.queue.writeBuffer(frameUniformBuffer, 0, frameArrayBuffer);

    const aspect = canvas.width / canvas.height;
    // Using mat4.perspectiveZO instead of mat4.perpective because WebGPU's
    // normalized device coordinates Z range is [0, 1], instead of WebGL's [-1, 1]
    mat4.perspectiveZO(projectionMatrix, Math.PI * 0.5, aspect, 0.01, 4.0);
    const commandEncoder = device.createCommandEncoder();
    colorAttachment.view = context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [colorAttachment],
      depthStencilAttachment
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
        for (const primitive of primitives) {
          for (const [bufferIndex, buffer] of Object.entries(primitive.buffers)) {
            renderPass.setVertexBuffer(Number.parseInt(bufferIndex), buffer.buffer, buffer.offset);
          }
          if (primitive.indexBuffer) {
            renderPass.setIndexBuffer(primitive.indexBuffer, primitive.indexType, primitive.indexOffset);
          }

          if (primitive.indexBuffer) {
            renderPass.drawIndexed(primitive.drawCount, primitive.instances.count, 0, 0, primitive.instances.first);
          } else {
            renderPass.draw(primitive.drawCount, primitive.instances.count, 0, primitive.instances.first);
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

function orbitCamera(element, target, distance, callback) {
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
    element.removeEventListener("pointermove", pointerMove);
    element.removeEventListener("pointerup", pointerUp);
  }

  element.addEventListener("pointerdown", event => {
    element.addEventListener("pointermove", pointerMove);
    element.addEventListener("pointerup", pointerUp);
  });

  element.addEventListener("mousewheel", event => {
    const wheelDeltaY = event.wheelDeltaY;
    if (wheelDeltaY) {
      distance -= wheelDeltaY * 0.005;
      broadcast();
    }
    event.preventDefault();
  });

}
