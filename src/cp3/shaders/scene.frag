#version 330 core
in vec3 vNormal;
in vec3 vColor;

uniform vec3 uLightDirection;
uniform vec3 uEyePosition;
uniform vec3 uTint;

out vec4 FragColor;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 light = normalize(-uLightDirection);
    float diffuse = max(dot(normal, light), 0.0);
    float ambient = 0.30;
    vec3 viewDirection = normalize(uEyePosition);
    float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 2.0) * 0.07;
    vec3 color = vColor * uTint * (ambient + diffuse * 0.78) + rim;
    FragColor = vec4(color, 1.0);
}
