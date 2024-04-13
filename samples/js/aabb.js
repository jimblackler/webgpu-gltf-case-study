import {mat4, vec3} from '../node_modules/gl-matrix/esm/index.js';

// Very simple AABB tracking so that we can position cameras sensibly.
export class AABB {
  min = vec3.fromValues(Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
  max = vec3.fromValues(Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);

  constructor(aabb) {
    if (aabb) {
      vec3.copy(this.min, aabb.min);
      vec3.copy(this.max, aabb.max);
    }
  }

  union(other) {
    vec3.min(this.min, this.min, other.min);
    vec3.max(this.max, this.max, other.max);
  }

  transform(mat) {
    const corners = [
      [this.min[0], this.min[1], this.min[2]],
      [this.min[0], this.min[1], this.max[2]],
      [this.min[0], this.max[1], this.min[2]],
      [this.min[0], this.max[1], this.max[2]],
      [this.max[0], this.min[1], this.min[2]],
      [this.max[0], this.min[1], this.max[2]],
      [this.max[0], this.max[1], this.min[2]],
      [this.max[0], this.max[1], this.max[2]],
    ];

    vec3.set(this.min, Number.MAX_VALUE, Number.MAX_VALUE, Number.MAX_VALUE);
    vec3.set(this.max, Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE);

    for (const corner of corners) {
      vec3.transformMat4(corner, corner, mat);
      vec3.min(this.min, this.min, corner);
      vec3.max(this.max, this.max, corner);
    }
  }

  get center() {
    return vec3.fromValues(
        ((this.max[0] + this.min[0]) * 0.5),
        ((this.max[1] + this.min[1]) * 0.5),
        ((this.max[2] + this.min[2]) * 0.5),
    );
  }

  get radius() {
    return vec3.distance(this.max, this.min) * 0.5;
  }
}
