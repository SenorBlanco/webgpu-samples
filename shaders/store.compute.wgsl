@group(0) @binding(0) var tex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1, 1, 1)
fn main() {
  for (var y = 0; y < 100; y++) {
    for (var x = 0; x < 100; x++) {
      textureStore(tex, vec2(x, y), vec4(f32(x) / 100.0, f32(y) / 100.0, 0.0, 1.0));
    }
  }
}
