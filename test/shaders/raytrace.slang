struct Payload {
    float3 color;
};

struct PushConstants {
    uint3 imageSize;
};

[[vk::push_constant]]
PushConstants pc;

[shader("raygeneration")]
void rayGenMain(
    RaytracingAccelerationStructure scene : register(t0, space0),
    RWTexture2D<float4> storageImage : register(u0, space1)) {
    Payload payload;
    payload.color = float3(0, 0, 0);

    RayDesc ray;
    ray.Origin = float3(0, 1.0, -3.0);
    ray.Direction = normalize(float3((DispatchRaysIndex().xy / float2(pc.imageSize.xy)) * 2.0 - 1.0, 1.0));
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
    uint2 pixel = uint2(DispatchRaysIndex().x, pc.imageSize.y - DispatchRaysIndex().y);
    storageImage[pixel] = float4(payload.color, 1.0);
}

[shader("miss")]
void missMain(inout Payload payload) {
    payload.color = float3(0.3, 0.3, 0.3); // Grey for missed rays
}

[shader("closesthit")]
void closestHitMain(inout Payload payload) {
    float3 hitPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.color = hitPosition.xzy * 0.5 + 0.5;
}

[shader("anyhit")]
void anyHitMain(inout Payload payload) {
    // empty
}
