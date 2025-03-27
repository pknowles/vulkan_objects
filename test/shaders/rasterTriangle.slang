// Copyright (c) 2025 Pyarelal Knowles, MIT License

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vsMain(uint vertexIndex : SV_VertexID)
{
    static const float3 vertices[3] = {
        float3(0.0,  0.5, 0.0),  // Top vertex
        float3(-0.5, -0.5, 0.0), // Bottom-left vertex
        float3(0.5, -0.5, 0.0)   // Bottom-right vertex
    };
    static const float4 colors[3] = {
        float4(1.0, 0.0, 0.0, 1.0), // Red
        float4(0.0, 1.0, 0.0, 1.0), // Green
        float4(0.0, 0.0, 1.0, 1.0)  // Blue
    };

    VertexOutput output;
    output.position = float4(vertices[vertexIndex], 1.0);
    output.color = colors[vertexIndex];
    return output;
}

[shader("pixel")]
float4 psMain(VertexOutput input) : SV_Target
{
    return input.color;
}
