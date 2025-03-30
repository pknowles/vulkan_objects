// Common Payload
struct Payload {
    float3 color;
};

// Ray Generation Shader
[shader("raygeneration")]
void rayGenMain(
    RaytracingAccelerationStructure scene : register(t0),
    RWTexture2D<float4> storageImage : register(u0, space1)) {
    Payload payload;
    payload.color = float3(0, 0, 0);

    RayDesc ray;
    ray.Origin = float3(0, 2.0, -5.0);
    ray.Direction = normalize(float3((DispatchRaysIndex().xy / 1024.0) * 2.0 - 1.0, 1.0));
    ray.TMin = 0.0;
    ray.TMax = 1000.0;

    TraceRay(
        scene,                          // Acceleration structure
        RAY_FLAG_NONE,                  // Ray flags
        0xFF,                           // Ray mask
        0,                              // SBT record offset
        1,                              // SBT record stride
        0,                              // Miss index
        ray,                            // Ray description
        payload);                       // Payload

    // Write the color to the first storage image
    storageImage[DispatchRaysIndex().xy] = float4(payload.color, 1.0);
}

// Miss Shader
[shader("miss")]
void missMain(inout Payload payload) {
    payload.color = float3(0.3, 0.3, 0.3); // Grey for missed rays
}

// Closest Hit Shader
[shader("closesthit")]
void closestHitMain(inout Payload payload) {
    payload.color = float3(1.0, 0.0, 0.0); // Red for hits
}

// Any-Hit Shader (optional)
[shader("anyhit")]
void anyHitMain(inout Payload payload) {
    // Example logic: ignore hits with a certain condition
    //if (payload.color.g > 0.5) // Example condition
    //    AcceptHitAndEndSearch();
    payload.color = float3(1.0, 1.0, 0.0); // Red for hits
}
