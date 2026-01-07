# Team Project - Group 7
## Group Members
- Corentin JEANNE \<cjeanne@kth.se>
- Timoteo SMART \<timoteo@ug.kth.se>
- Midhushaan Ananthakumar \<midana@kth.se>

## Compiling
Inside the `src` directory, execute
```bash
make [DEFINES=DEFINES]
```
Where `DEFINES` is a space separated list the the [possible optimisations](#possible-optimizations). Surround the list in quotes `"` if needed, e.g.
```
make DEFINES="BOUNCES=10 SMART_BOUNCES"
```

## Profiling and Testing
Use the `run` script at the root of th project to quickly test optimisations. **You do not need to compile the code beforehand to run the script**.
```
./run OPTIMIZATIONS
```

Where `OPTIMIZATIONS` is a space separated list the the [possible optimisations](#possible-optimizations). Surround the list in quotes `"` if needed, e.g.
```
./run "BOUNCES=10 SMART_BOUNCES"
```
The following aliases are also available when executing `run`
- `BEST`: `FAST_MATH BOUNCES=10 SMART_BOUNCES BVH THREADS_Y=4`
- `BASELINE`

The script will generated 3 files in the `results` folder:
- `OPTIMIZATIONS.log`: the output of the program, including the render time
- `OPTIMIZATIONS.ppm`: the rendered frame in a bitmap format. Use image editing tools such as GIMP to view it.
- `OPTIMIZATIONS.nsys-rep`: the NSight profiling data. It can be visualized with `nsys-ui`.

## Possible Optimizations
Every optimization is surrounded by a `#ifdef` statement and has its related define:
- `BOUNCES=N`: number of bounces
- `SMART_BOUNCES`: stop bouncing early
- `FAST_MATH`: enables the `--use-fast-math` compiler flag
- `OPTIMIZED_RENDER`: runs the `render` kernel 5 times and averages the results ind stead of casting 5 rays per pixels
- `PARALLEL_RAYS`: spawns 5 times as many threads and cast 1 rays per thread
- `RAYS_PER_PIXEL=N`: number of rays per pixel
- `THREADS_X=N`: threads per block on the x-axis
- `THREADS_Y=N`: threads per block on the y-axis
- `BVH`: replaces the naive hit-detections with a Bounding Volume Hierarchy
- `LENGTH_CACHING`: cache the length of `vec3`
- `REDUCED_PRECISION`: use `fp16` when possible