/**
 * TinyGltf
 * Loads glTF 2.0 file, resolves buffer and image dependencies, and computes node world transforms.
 * This is a VERY simplified glTF loader that avoids doing too much work for you.
 * It should generally not be used outside of simple tutorials or examples.
 */

import {mat4} from "../node_modules/gl-matrix/esm/index.js";

const GLB_MAGIC = 0x46546C67;
const JSON_CHUNK_TYPE = 0x4E4F534A;
const BIN_CHUNK_TYPE = 0x004E4942;

export function getWorldMatrixMap(gltf) {
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
              node.rotation,
              node.translation,
              node.scale);
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

export class TinyGltf {
  constructor(device) {
    this.device = device;
    this.defaultSampler = createGpuSamplerFromSampler(device);
  }

  async loadFromUrl(url) {
    const i = url.lastIndexOf("/");
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

    // Buffers will be exposed as ArrayBuffers.
    // Images will be exposed as ImageBitmaps.

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
          {type: image.mimeType})).then(image => createGpuTextureFromImage(this.device, image))
    }));

    const gpuSamplers = Object.values(gltf.samplers ?? []).map(sampler =>
        createGpuSamplerFromSampler(this.device, sampler));

    return {
      gltf,
      gpuBuffers: Object.values(gltf.bufferViews).map((bufferView, index) =>
          createGpuBufferFromBufferView(this.device, bufferView,
              chunks[BIN_CHUNK_TYPE], bufferViewUsages[index])),
      gpuTextures: Object.values(gltf.textures ?? []).map(texture => ({
        texture: imageTextures[texture.source],
        sampler: texture.sampler ? gpuSamplers[texture.sampler] : this.defaultSampler
      })),
      gpuDefaultSampler: this.defaultSampler
    }
  }

  static componentCountForType(type) {
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


}

/**
 * TinyGltfWebGPU
 * Loads glTF 2.0 file and creates the necessary WebGPU buffers, textures, and samplers for you.
 * As with the base TinyGltf, this is a VERY simplified loader and should not be used outside of
 * simple tutorials or examples.
 */

// To make it easier to reference the WebGL enums that glTF uses.
const GL = WebGLRenderingContext;

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

function createGpuSamplerFromSampler(device, sampler = {name: "glTF default sampler"}) {
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
