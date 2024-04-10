// This file contains the shared source for our demo app, handling the basics of model loading and
// the associated picker UI, camera positioning, and the render loop.

import { TinyWebGpuDemo } from './tiny-webgpu-demo.js'
import { TinyGltfWebGpu } from './tiny-gltf.js'
import { QueryArgs } from './query-args.js'

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

// Runs the basic render loop, model switching, and camera handling.
export class GltfDemo extends TinyWebGpuDemo {
  clearColor = {r: 0.0, g: 0.0, b: 0.2, a: 1.0};

  rendererClass = null;
  gltfRenderer = null;

  constructor(rendererClass, startup_model) {
    super();

    // Allow the startup model to be overriden by a query arg.
    startup_model = QueryArgs.getString('model', startup_model);
    if (startup_model in GltfModels) {
      this.model = GltfModels[startup_model];
    } else {
      this.model = GltfModels['antique_camera'];
    }

    this.rendererClass = rendererClass;

    this.pane.addBlade({
      label: 'model',
      view: 'list',
      options: GltfModels,
      value: this.model,
    }).on('change', (ev) => {
      this.onLoadModel(this.device, ev.value);
    });
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

      this.gltfRenderer = new this.rendererClass(this, gltf);

      device.popErrorScope().then((error) => {
        this.setError(error, 'loading glTF model');
        if (error) {
          this.gltfRenderer = null;
        }
      });
    } catch(error) {
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
}
