import { mat4, vec3, vec4 } from 'wgpu-matrix';
import { GUI } from 'dat.gui';
import { mesh } from '../../meshes/stanfordDragon';

// LightData
struct LightData {
  position : vec4f,
  color : vec3f,
  radius : f32,
}
struct LightsBuffer {
  lights: array<LightData>,
}

// Config
struct Config {
  numLights : u32,
}

// LightExtent
struct LightExtent {
  min : vec4f,
  max : vec4f,
}

// LightUpdateBindings
@group(0) @binding(1) var<uniform> config: Config;
@group(0) @binding(0) var<storage, read_write> lightsBuffer: LightsBuffer;
@group(0) @binding(2) var<uniform> lightExtent: LightExtent;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3u) {
    var i = GlobalInvocationID.x;
    if (i >= config.numLights) {
      return;
    }

    lightsBuffer.lights[i].position.y = lightsBuffer.lights[i].position.y - 0.5 - 0.003 * (f32(i) - 64.0 * floor(f32(i) / 64.0));

    if (lightsBuffer.lights[i].position.y < lightExtent.min.y) {
        lightsBuffer.lights[i].position.y = lightExtent.max.y;
    }
}

// Uniforms
struct Uniforms {
  modelMatrix : mat4x4f,
  normalModelMatrix : mat4x4f,
}

// Camera
struct Camera {
  viewProjectionMatrix : mat4x4f,
  invViewProjectionMatrix : mat4x4f,
}

// VertexOutput
struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragNormal: vec3f,    // normal in world space
  @location(1) fragUV: vec2f,
}

// WriteGBuffersBindings
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<uniform> camera : Camera;

struct GBufferOutput {
  @location(0) normal : vec4f,
  @location(1) albedo : vec4f,
}

@vertex
fn main(
// Vertex
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(2) uv : vec2f
) -> VertexOutput {
  // WriteGBuffers vertex shader

    // Transform the vertex position by the model and viewProjection matrices.
    // Transform the vertex normal by the normalModelMatrix (inverse transpose of the model).
    var output : VertexOutput;
    let worldPosition = (uniforms.modelMatrix * vec4(position, 1.0)).xyz;
    output.Position = camera.viewProjectionMatrix * vec4(worldPosition, 1.0);
    output.fragNormal = normalize((uniforms.normalModelMatrix * vec4(normal, 1.0)).xyz);
    output.fragUV = uv;
    return output;
}

@fragment
fn main(
  @location(0) fragNormal: vec3f,
  @location(1) fragUV : vec2f
) -> GBufferOutput {
    // WriteGBuffers fragment shader

    let uv = floor(30.0 * fragUV);
    let c = 0.2 + 0.5 * ((uv.x + uv.y) - 2.0 * floor((uv.x + uv.y) / 2.0));

    var output : GBufferOutput;
    output.normal = vec4(normalize(fragNormal), 1.0);
    output.albedo = vec4(c, c, c, 1.0);

    return output;
}

@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4f {
    // TextureQuadPass vertex shader
    const pos = array(
      vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
      vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    );

    return vec4f(pos[VertexIndex], 0.0, 1.0);
}

// GBufferTextureBindings
@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_2d<f32>;

// CanvasSizeBindings
override canvasSizeWidth: f32;
override canvasSizeHeight: f32;

// GBufferTextureBindings
@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_2d<f32>;

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
    var result : vec4f;
    let c = coord.xy / vec2f(canvasSizeWidth, canvasSizeHeight);
    if (c.x < 0.33333) {
      let rawDepth = textureLoad(gBufferDepth, vec2i(floor(coord.xy)), 0).x;
      // remap depth into something a bit more visible
      let depth = (1.0 - rawDepth) * 50.0;
      result = vec4(depth);
    } else if (c.x < 0.66667) {
      result = textureLoad(gBufferNormal, vec2i(floor(coord.xy)), 0);
      result.x = (result.x + 1.0) * 0.5;
      result.y = (result.y + 1.0) * 0.5;
      result.z = (result.z + 1.0) * 0.5;
    } else {
      result = textureLoad(gBufferAlbedo, vec2i(floor(coord.xy)), 0);
    }
    return result;
}

struct LightData {
  position : vec4f,
  color : vec3f,
  radius : f32,
}
struct LightsBuffer {
  lights: array<LightData>,
}


struct Config {
  numLights : u32,
}
struct Camera {
  viewProjectionMatrix : mat4x4f,
  invViewProjectionMatrix : mat4x4f,
}

// DeferredRenderBufferBindings
@group(1) @binding(0) var<storage, read> lightsBuffer: LightsBuffer;
@group(1) @binding(1) var<uniform> config: Config;
@group(1) @binding(2) var<uniform> camera: Camera;

  // worldFromScreenCoord
fn world_from_screen_coord(coord : vec2f, depth_sample: f32) -> vec3f {
  // reconstruct world-space position from the screen coordinate.
  let posClip = vec4(coord.x * 2.0 - 1.0, (1.0 - coord.y) * 2.0 - 1.0, depth_sample, 1.0);
  let posWorldW = camera.invViewProjectionMatrix * posClip;
  let posWorld = posWorldW.xyz / posWorldW.www;
  return posWorld;
}

  // DeferredRender fragment shader
@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
    var result : vec3f;

    // Retrieve the depth from the depth buffer.
    let depth = textureLoad(gBufferDepth, vec2i(floor(coord.xy)), 0).x;

    // Don't light the sky.
    if (depth >= 1.0) {
      discard;
    }

    let bufferSize = textureDimensions(gBufferDepth);
    let coordUV = coord.xy / vec2f(bufferSize);
    let position = world_from_screen_coord(coordUV, depth);
    let normal = textureLoad(gBufferNormal, vec2i(floor(coord.xy)), 0).xyz;
    let albedo = textureLoad(gBufferAlbedo, vec2i(floor(coord.xy)), 0).rgb;

    for (var i = 0u; i < config.numLights; i++) {
      let L = lightsBuffer.lights[i].position.xyz - position;
      let distance = length(L);
      if (distance > lightsBuffer.lights[i].radius) {
        continue;
      }
      let lambert = max(dot(normal, normalize(L)), 0.0);
      result += vec3f(
        lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * lightsBuffer.lights[i].color * albedo
      );
    }

    // some manual ambient
    result += vec3(0.2);

    return vec4(result, 1.0);
}
import { quitIfWebGPUNotAvailable, quitIfLimitLessThan } from '../util';

// Host code

const kMaxNumLights = 1024;
const lightExtentMin = vec3.fromValues(-50, -30, -50);
const lightExtentMax = vec3.fromValues(50, 50, 50);

const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const adapter = await navigator.gpu?.requestAdapter({
  featureLevel: 'compatibility',
});
const limits: Record<string, GPUSize32> = {};
quitIfLimitLessThan(adapter, 'maxStorageBuffersInFragmentStage', 1, limits);
const device = await adapter?.requestDevice({
  requiredLimits: limits,
});
quitIfWebGPUNotAvailable(adapter, device);

const context = canvas.getContext('webgpu') as GPUCanvasContext;

const devicePixelRatio = window.devicePixelRatio;
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
const aspect = canvas.width / canvas.height;
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device,
  format: presentationFormat,
});

// Create the model vertex buffer.
const kVertexStride = 8;
const vertexBuffer = device.createBuffer({
  // position: vec3, normal: vec3, uv: vec2
  size: mesh.positions.length * kVertexStride * Float32Array.BYTES_PER_ELEMENT,
  usage: GPUBufferUsage.VERTEX,
  mappedAtCreation: true,
});
{
  const mapping = new Float32Array(vertexBuffer.getMappedRange());
  for (let i = 0; i < mesh.positions.length; ++i) {
    mapping.set(mesh.positions[i], kVertexStride * i);
    mapping.set(mesh.normals[i], kVertexStride * i + 3);
    mapping.set(mesh.uvs[i], kVertexStride * i + 6);
  }
  vertexBuffer.unmap();
}

// Create the model index buffer.
const indexCount = mesh.triangles.length * 3;
const indexBuffer = device.createBuffer({
  size: indexCount * Uint16Array.BYTES_PER_ELEMENT,
  usage: GPUBufferUsage.INDEX,
  mappedAtCreation: true,
});
{
  const mapping = new Uint16Array(indexBuffer.getMappedRange());
  for (let i = 0; i < mesh.triangles.length; ++i) {
    mapping.set(mesh.triangles[i], 3 * i);
  }
  indexBuffer.unmap();
}

// Create normals texture
const gBufferTexture2DFloat16 = device.createTexture({
  size: [canvas.width, canvas.height],
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  format: 'rgba16float',
});

// Create albedo texture
const gBufferTextureAlbedo = device.createTexture({
  size: [canvas.width, canvas.height],
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  format: 'bgra8unorm',
});

// Create depth texture
const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: 'depth24plus',
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

const gBufferTextureViews = [
  gBufferTexture2DFloat16.createView(),
  gBufferTextureAlbedo.createView(),
  depthTexture.createView(),
];

// Vertex layout

const vertexBuffers: Iterable<GPUVertexBufferLayout> = [
  {
    arrayStride: Float32Array.BYTES_PER_ELEMENT * 8,
    attributes: [
      {
        // position
        shaderLocation: 0,
        offset: 0,
        format: 'float32x3',
      },
      {
        // normal
        shaderLocation: 1,
        offset: Float32Array.BYTES_PER_ELEMENT * 3,
        format: 'float32x3',
      },
      {
        // uv
        shaderLocation: 2,
        offset: Float32Array.BYTES_PER_ELEMENT * 6,
        format: 'float32x2',
      },
    ],
  },
];

const primitive: GPUPrimitiveState = {
  topology: 'triangle-list',
  cullMode: 'back',
};

// Create WriteGBuffers RenderPipeline
const writeGBuffersPipeline = device.createRenderPipeline({
  layout: 'auto',
  vertex: {
    module: device.createShaderModule({
      code: vertexWriteGBuffers,
    }),
    buffers: vertexBuffers,
  },
  fragment: {
    module: device.createShaderModule({
      code: fragmentWriteGBuffers,
    }),
    targets: [
      // normal
      { format: 'rgba16float' },
      // albedo
      { format: 'bgra8unorm' },
    ],
  },
  depthStencil: {
    depthWriteEnabled: true,
    depthCompare: 'less',
    format: 'depth24plus',
  },
  primitive,
});

const gBufferTexturesBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT,
      texture: {
        sampleType: 'unfilterable-float',
      },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT,
      texture: {
        sampleType: 'unfilterable-float',
      },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.FRAGMENT,
      texture: {
        sampleType: 'unfilterable-float',
      },
    },
  ],
});

const lightsBufferBindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: {
        type: 'read-only-storage',
      },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: {
        type: 'uniform',
      },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: {
        type: 'uniform',
      },
    },
  ],
});

// Create GBuffersDebugView RenderPipeline

const gBuffersDebugViewPipeline = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [gBufferTexturesBindGroupLayout],
  }),
  vertex: {
    module: device.createShaderModule({
      code: vertexTextureQuad,
    }),
  },
  fragment: {
    module: device.createShaderModule({
      code: fragmentGBuffersDebugView,
    }),
    targets: [
      {
        format: presentationFormat,
      },
    ],
    constants: {
      canvasSizeWidth: canvas.width,
      canvasSizeHeight: canvas.height,
    },
  },
  primitive,
});

// Create DeferredRender RenderPipeline

const deferredRenderPipeline = device.createRenderPipeline({
  layout: device.createPipelineLayout({
    bindGroupLayouts: [
      gBufferTexturesBindGroupLayout,
      lightsBufferBindGroupLayout,
    ],
  }),
  vertex: {
    module: device.createShaderModule({
      code: vertexTextureQuad,
    }),
  },
  fragment: {
    module: device.createShaderModule({
      code: fragmentDeferredRendering,
    }),
    targets: [
      {
        format: presentationFormat,
      },
    ],
  },
  primitive,
});

// Create ColorAttachments for GBuffer textures
const writeGBufferPassDescriptor: GPURenderPassDescriptor = {
  colorAttachments: [
    {
      view: gBufferTextureViews[0],

      clearValue: [0.0, 0.0, 1.0, 1.0],
      loadOp: 'clear',
      storeOp: 'store',
    },
    {
      view: gBufferTextureViews[1],

      clearValue: [0, 0, 0, 1],
      loadOp: 'clear',
      storeOp: 'store',
    },
  ],
  depthStencilAttachment: {
    view: depthTexture.createView(),

    depthClearValue: 1.0,
    depthLoadOp: 'clear',
    depthStoreOp: 'store',
  },
};

const textureQuadPassDescriptor: GPURenderPassDescriptor = {
  colorAttachments: [
    {
      // view is acquired and set in render loop.
      view: undefined,

      clearValue: [0, 0, 0, 1],
      loadOp: 'clear',
      storeOp: 'store',
    },
  ],
};

// Settings
const settings = {
  mode: 'rendering',
  numLights: 128,
};

// Create Config uniform buffer
const configUniformBuffer = (() => {
  const buffer = device.createBuffer({
    size: Uint32Array.BYTES_PER_ELEMENT,
    mappedAtCreation: true,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  new Uint32Array(buffer.getMappedRange())[0] = settings.numLights;
  buffer.unmap();
  return buffer;
})();

const gui = new GUI();
gui.add(settings, 'mode', ['rendering', 'gBuffers view']);
gui
  .add(settings, 'numLights', 1, kMaxNumLights)
  .step(1)
  .onChange(() => {
    device.queue.writeBuffer(
      configUniformBuffer,
      0,
      new Uint32Array([settings.numLights])
    );
  });

// Create Uniforms uniform buffer
const modelUniformBuffer = device.createBuffer({
  size: 4 * 16 * 2, // two 4x4 matrix
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// Create Camera uniform buffer
const cameraUniformBuffer = device.createBuffer({
  size: 4 * 16 * 2, // two 4x4 matrix
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

// Create WriteGBuffersBindings BindGroup
const sceneUniformBindGroup = device.createBindGroup({
  layout: writeGBuffersPipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: modelUniformBuffer,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: cameraUniformBuffer,
      },
    },
  ],
});

// Create GBufferTextureBindings BindGroup
const gBufferTexturesBindGroup = device.createBindGroup({
  layout: gBufferTexturesBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: gBufferTextureViews[0],
    },
    {
      binding: 1,
      resource: gBufferTextureViews[1],
    },
    {
      binding: 2,
      resource: gBufferTextureViews[2],
    },
  ],
});

// Lights data are uploaded in a storage buffer
// which could be updated/culled/etc. with a compute shader
const extent = vec3.sub(lightExtentMax, lightExtentMin);

// Create storage buffer for lights
const lightDataStride = 8;
const bufferSizeInByte =
  Float32Array.BYTES_PER_ELEMENT * lightDataStride * kMaxNumLights;
const lightsBuffer = device.createBuffer({
  size: bufferSizeInByte,
  usage: GPUBufferUsage.STORAGE,
  mappedAtCreation: true,
});

// We randomaly populate lights randomly in a box range
// And simply move them along y-axis per frame to show they are
// dynamic lightings
const lightData = new Float32Array(lightsBuffer.getMappedRange());
const tmpVec4 = vec4.create();
let offset = 0;
for (let i = 0; i < kMaxNumLights; i++) {
  offset = lightDataStride * i;
  // position
  for (let i = 0; i < 3; i++) {
    tmpVec4[i] = Math.random() * extent[i] + lightExtentMin[i];
  }
  tmpVec4[3] = 1;
  lightData.set(tmpVec4, offset);
  // color
  tmpVec4[0] = Math.random() * 2;
  tmpVec4[1] = Math.random() * 2;
  tmpVec4[2] = Math.random() * 2;
  // radius
  tmpVec4[3] = 20.0;
  lightData.set(tmpVec4, offset + 4);
}
lightsBuffer.unmap();

const lightExtentBuffer = device.createBuffer({
  size: 4 * 8,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const lightExtentData = new Float32Array(8);
lightExtentData.set(lightExtentMin, 0);
lightExtentData.set(lightExtentMax, 4);
device.queue.writeBuffer(
  lightExtentBuffer,
  0,
  lightExtentData.buffer,
  lightExtentData.byteOffset,
  lightExtentData.byteLength
);

// Create LightUpdate ComputePipeline
const lightUpdateComputePipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: device.createShaderModule({
      code: lightUpdate,
    }),
  },
});

// Create DeferredRenderBufferBindings BindGroup
const lightsBufferBindGroup = device.createBindGroup({
  layout: lightsBufferBindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: {
        buffer: lightsBuffer,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: configUniformBuffer,
      },
    },
    {
      binding: 2,
      resource: {
        buffer: cameraUniformBuffer,
      },
    },
  ],
});

// Create LightUpdateBindings BindGroup
const lightsBufferComputeBindGroup = device.createBindGroup({
  layout: lightUpdateComputePipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: lightsBuffer,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: configUniformBuffer,
      },
    },
    {
      binding: 2,
      resource: {
        buffer: lightExtentBuffer,
      },
    },
  ],
});
//--------------------

// Scene matrices
const eyePosition = vec3.fromValues(0, 50, -100);
const upVector = vec3.fromValues(0, 1, 0);
const origin = vec3.fromValues(0, 0, 0);

const projectionMatrix = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 2000.0);

// Move the model so it's centered.
const modelMatrix = mat4.translation([0, -45, 0]);

// Compute the inverse transpose for transforming normals.
const invertTransposeModelMatrix = mat4.invert(modelMatrix);
mat4.transpose(invertTransposeModelMatrix, invertTransposeModelMatrix);

// Set the matrix uniform data.
const normalModelData = invertTransposeModelMatrix;
device.queue.writeBuffer(modelUniformBuffer, 0, modelMatrix);
device.queue.writeBuffer(
  modelUniformBuffer,
  64,
  normalModelData.buffer,
  normalModelData.byteOffset,
  normalModelData.byteLength
);

function getCameraViewProjMatrix() {
  // Rotates the camera around the origin based on time.
  const rad = Math.PI * (Date.now() / 5000);
  const rotation = mat4.rotateY(mat4.translation(origin), rad);
  const rotatedEyePosition = vec3.transformMat4(eyePosition, rotation);

  const viewMatrix = mat4.lookAt(rotatedEyePosition, origin, upVector);

  return mat4.multiply(projectionMatrix, viewMatrix);
}

function frame() {
  // Update camera matrices
  const cameraViewProj = getCameraViewProjMatrix();
  device.queue.writeBuffer(
    cameraUniformBuffer,
    0,
    cameraViewProj.buffer,
    cameraViewProj.byteOffset,
    cameraViewProj.byteLength
  );
  const cameraInvViewProj = mat4.invert(cameraViewProj);
  device.queue.writeBuffer(
    cameraUniformBuffer,
    64,
    cameraInvViewProj.buffer,
    cameraInvViewProj.byteOffset,
    cameraInvViewProj.byteLength
  );

  const commandEncoder = device.createCommandEncoder();
  {
    // Write position, normal, albedo etc. data to gBuffers
    const gBufferPass = commandEncoder.beginRenderPass(
      writeGBufferPassDescriptor
    );
    gBufferPass.setPipeline(writeGBuffersPipeline);
    gBufferPass.setBindGroup(0, sceneUniformBindGroup);
    gBufferPass.setVertexBuffer(0, vertexBuffer);
    gBufferPass.setIndexBuffer(indexBuffer, 'uint16');
    gBufferPass.drawIndexed(indexCount);
    gBufferPass.end();
  }
  {
    // Update lights position
    const lightPass = commandEncoder.beginComputePass();
    lightPass.setPipeline(lightUpdateComputePipeline);
    lightPass.setBindGroup(0, lightsBufferComputeBindGroup);
    lightPass.dispatchWorkgroups(Math.ceil(kMaxNumLights / 64));
    lightPass.end();
  }
  if (settings.mode === 'gBuffers view') {
    // GBuffers debug view
    // Left: depth
    // Middle: normal
    // Right: albedo (use uv to mimic a checkerboard texture)
    textureQuadPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();
    const debugViewPass = commandEncoder.beginRenderPass(
      textureQuadPassDescriptor
    );
    debugViewPass.setPipeline(gBuffersDebugViewPipeline);
    debugViewPass.setBindGroup(0, gBufferTexturesBindGroup);
    debugViewPass.draw(6);
    debugViewPass.end();
  } else {
    // Deferred rendering
    textureQuadPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();
    const deferredRenderingPass = commandEncoder.beginRenderPass(
      textureQuadPassDescriptor
    );
    deferredRenderingPass.setPipeline(deferredRenderPipeline);
    deferredRenderingPass.setBindGroup(0, gBufferTexturesBindGroup);
    deferredRenderingPass.setBindGroup(1, lightsBufferBindGroup);
    deferredRenderingPass.draw(6);
    deferredRenderingPass.end();
  }
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
