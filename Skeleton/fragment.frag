		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
			vec3 dir;
			float fov;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 ka = material.ka;
			vec3 kd = material.kd;
			
			vec3 radiance = vec3(0,0,0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);			
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				if( dot(lights[i].dir, wLight[i]) < lights[i].fov)
				{
					radiance += ka * lights[i].La + 
							((kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le);
				}
			}
			fragmentColor = vec4(radiance, 1);
		}