@group(0) @binding(0) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(1) var gBufferAlbedo: texture_2d<f32>;
@group(0) @binding(2) var gBufferDepth: texture_2d<f32>;

override canvasSizeWidth: f32;
override canvasSizeHeight: f32;

@fragment
fn main(
  @builtin(position) coord : vec4f
) -> @location(0) vec4f {
  var result : vec4f;
  let c = coord.xy / vec2f(canvasSizeWidth, canvasSizeHeight);
  if (c.x < 0.33333) {
    let rawDepth = textureLoad(
      gBufferDepth,
      vec2i(floor(coord.xy)),
      0
    ).x;
    // remap depth into something a bit more visible
    let depth = (1.0 - rawDepth) * 50.0;
    result = vec4(depth);
  } else if (c.x < 0.66667) {
    result = textureLoad(
      gBufferNormal,
      vec2i(floor(coord.xy)),
      0
    );
    result.x = (result.x + 1.0) * 0.5;
    result.y = (result.y + 1.0) * 0.5;
    result.z = (result.z + 1.0) * 0.5;
  } else {
    result = textureLoad(
      gBufferAlbedo,
      vec2i(floor(coord.xy)),
      0
    );
  }
  return result;
}
