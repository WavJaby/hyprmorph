# hyprmorph

A Hyprland plugin that redraws window resize animations so the content keeps its layout while the window box animates.

![Discord resized with Hyprland's default animation (left) and with hyprmorph (right), slowed 4x](https://github.com/WavJaby/hyprmorph/releases/download/demo/demo.gif)

Natively, a window that changes size shows its old buffer stretched or cropped until the client sends a buffer at the new size, then jumps to it. This plugin instead:

1. holds the size/position animation at its start for up to `hold_ms` while it prepares the transition;
2. compares the old content with the client's first new-size frame in a separate worker process and builds a mesh of regions: rigid ones (text, icons, avatars) keep their scale and move, flexible ones (backgrounds, list rows, gaps) stretch;
3. animates both images along that mesh with the window box and crossfades old to new in the middle of the motion.

When no mesh is ready in time (slow client, refused analysis, animation already running), the window falls back to a plain scale of the old and new images with the same crossfade, never to the native stretch/crop.

## Status

First release. Tested on one machine: Hyprland 0.57.2, hyprutils 0.14.3, Arch Linux, Intel Arc (xe driver), headless and live outputs at scales 1 and 1.6. Expect rough edges on other hardware.

A plugin runs inside the compositor: a crash in it takes down Hyprland and every open window with it. Save your work before trying it, and keep a TTY login at hand.

## Requirements

- Hyprland 0.57.2 (the plugin is built against its headers; other versions are untested and may not load)
- hyprutils 0.14.3 (the animation hold hooks one of its functions; see Limitations)
- libdrm, EGL, GLESv2, GBM
- OpenCV 5 (`opencv` on Arch): core, imgproc, imgcodecs, features, geometry
- A C++23 compiler

## Install

### hyprpm

```sh
hyprpm update
hyprpm add https://github.com/WavJaby/hyprmorph
hyprpm enable hyprmorph
```

Then load enabled plugins at start-up in `~/.config/hypr/hyprland.lua`:

```lua
hl.on("hyprland.start", function()
    hl.exec_cmd("hyprpm reload -n")
end)
```

### Manual

```sh
make -j$(nproc)
hyprctl plugin load "$PWD/build/hyprmorph.so"
```

`build/hyprmorph.so` is the whole install: the worker executable is linked into it.

## Configuration

The animation hold needs a **bezier** curve on window animations. Hyprland 0.57's example config uses a spring (`spring = "easy"`) for `windows`; with a spring the plugin still draws the transition but cannot hold the animation, so the mesh is rarely ready in time and you mostly see plain scaling. Switch the `windows` leaves to a bezier:

```lua
hl.curve("easeOutQuint", { type = "bezier", points = { { 0.23, 1 }, { 0.32, 1 } } })
hl.animation({ leaf = "windows",   enabled = true, speed = 4.79, bezier = "easeOutQuint" })
hl.animation({ leaf = "windowsIn", enabled = true, speed = 4.1,  bezier = "easeOutQuint", style = "popin 87%" })
```

Options, set like any plugin option:

```lua
hl.config({ plugin = { hyprmorph = { hold_ms = 300 } } })
```

| Option | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Draw window size changes through the plugin (`false` = native drawing) |
| `mesh` | `true` | Draw the prepared mesh when ready (`false` = plain scale only) |
| `hold_ms` | `300` | Max ms the size/position animation waits at its start for the mesh (`0` = never hold). A window whose mesh is ready sooner starts sooner |
| `crossfade_span` | `0.3` | Old-to-new crossfade length in transition progress, starting at the midpoint |
| `crossfade_blur_px` | `1.5` | Peak blur (Gaussian sigma, logical px) in the middle of the crossfade (`0` = off) |
| `crossfade_ms` | `100` | Crossfade length when the new-size buffer arrives after the crossfade window opened |
| `new_buffer_timeout_ms` | `300` | After the animation ends, how long to keep the scaled old image while waiting for the new-size buffer |
| `mesh_ramp_ms` | `80` | When content changes under a drawn mesh, ms to relax it back to plain scale |
| `readiness` | `true` | Detect when the client's new-size content has settled (`false` = no mesh) |
| `verbose` | `false` | Log per-resize events to the Hyprland log (warnings are always logged) |
| `prep_worker` | empty | Path of a prep-worker executable to run instead of the built-in one |
| `readiness_dump` | empty | Debugging: directory for per-resize readiness thumbnails |

## Inspecting

```sh
hyprctl hyprmorph
```

prints JSON counters: holds, mesh answers and their ready times, how each transition ended, and worker state (`path`, render `node`, restarts, last error).

## Limitations

- **Spring curves:** no hold (see Configuration); transitions mostly use plain scaling.
- **hyprutils pin:** the hold patches the call from hyprutils' animation stepping to `getPercent`. It refuses to hook any libhyprutils other than the one the plugin was built against; after a hyprutils upgrade, rebuild (`hyprpm update`).
- **Not drawn by the plugin (native drawing):** X11/XWayland windows, windows on rotated or transformed outputs.
- **Single-window screen sharing:** while a window that is being shared by itself resizes, the shared stream shows it hidden until the transition ends. Not tested.
- **Multi-GPU:** the worker runs on the GPU Hyprland renders with. Only single-GPU rendering was tested.
- **Input** follows the window's final geometry from the start of the resize, as natively; only the drawing is animated.
- **Cost:** each resize runs the analysis on a worker process (typically 80–320 ms of one core plus some GPU work). Battery impact has not been measured.

## License

MIT, see [LICENSE](LICENSE).
