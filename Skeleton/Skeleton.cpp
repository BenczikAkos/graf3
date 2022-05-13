//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Benczik Akos
// Neptun : JWCCFA
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { 
//---------------------------
	float f; 
	T d;  
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 50;


//---------------------------
struct Camera {
//---------------------------
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	void Animate(float dt) {
		vec4 wEye4 = vec4(wEye.x, wEye.y, wEye.z, 1);
		wEye4 = wEye4 * RotationMatrix(dt, vec3(0, 1, 0));
		wEye = vec3(wEye4.x, wEye4.y, wEye4.z);
	}
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
	bool alwaysShine = false;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;
	vec3 dir;
	float fov;
};


//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};


//---------------------------
class Shader : public GPUProgram {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
			vec3 dir;
			float fov;
		};

		uniform mat4  MVP, M, Minv; 
		uniform Light[8] lights;    
		uniform int   nLights;
		uniform vec3  wEye;         

		layout(location = 0) in vec3  vtxPos;            
		layout(location = 1) in vec3  vtxNorm;      	 

		out vec3 wNormal;		    
		out vec3 wView;             
		out vec3 wLight[8];		    

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	const char* fragmentSource = R"(
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
			bool alwaysShine;
		};

		uniform Material material;
		uniform Light[8] lights;     
		uniform int   nLights;

		in  vec3 wNormal;       
		in  vec3 wView;         
		in  vec3 wLight[8];     
		
        out vec4 fragmentColor; 

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	
			vec3 ka = material.ka;
			vec3 kd = material.kd;
			
			vec3 radiance = vec3(0,0,0);
			for(int i = 0; i < 2; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);			
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				if(dot(normalize(lights[i].dir), normalize(wLight[i])) < lights[i].fov || material.alwaysShine) {
					radiance += ka * lights[i].La + 
							((kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le);
				}
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	Shader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
		setUniform(material.alwaysShine, name + ".alwaysShine");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
		setUniform(light.dir, name + ".dir");
		setUniform(light.fov, name + ".fov");
	}
};


//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0); 
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

//---------------------------
class Cylinder : public ParamSurface {
	//---------------------------
public:
	Cylinder() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		X = Cos(U); Z = Sin(U); Y = V;
	}
};

//---------------------------
class Paraboloid : public ParamSurface {
	//---------------------------
public:
	Paraboloid() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = V * 2;
		X = Cos(U) * V; Z = Sin(U) * V; Y = Pow(V, 2);
	}
};

//---------------------------
class Circle : public ParamSurface {
	//---------------------------
public:
	Circle() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI;
		X = Cos(U) * V; Z = Sin(U) * V; Y = 0;
	}

};

//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	Object* child = nullptr;
	vec3 refPoint;
public:
	Object(Shader* _shader, Material* _material, Geometry* _geometry, vec3 translate) :
		scale(vec3(1, 1, 1)), translation(translate), rotationAxis(0, 1, 0), rotationAngle(7.7) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}

	void setChild(Object* object) {
		child = object;
		child->refPoint = child->refPoint + refPoint;
	}

	const vec3 getLastRefPoint() {
		if (child == nullptr)
			return refPoint;
		return child->getLastRefPoint();
	}

	void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) *
			RotationMatrix(rotationAngle, rotationAxis) *
			TranslateMatrix(translation) * state.M;
		vec4 refPointv4 = vec4(0, 0, 0, 1) * state.M;
		refPoint = vec3(refPointv4.x, refPointv4.y, refPointv4.z);
		state.Minv = state.Minv *
			TranslateMatrix(-translation) *
			RotationMatrix(-rotationAngle, rotationAxis) *
			ScaleMatrix(vec3(1/scale.x, 1/scale.y, 1/scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
		state.M = ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z)) * state.M;
		state.Minv = state.Minv * ScaleMatrix(scale);
		if (child != nullptr) {
			child->Draw(state);
		}

	}

	virtual void Animate(float tstart, float tend) { 
		rotationAngle += 0.5f * (tend - tstart);
		if (child != nullptr) {
			child->Animate(tstart, tend);
		}
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	Camera camera;
	std::vector<Light> lights;
public:
	void Build() {
		Shader* shader = new Shader();

		Material* matter = new Material;
		matter->kd = vec3(0.8f, 0.8f, 0.8f);
		matter->ks = vec3(4, 4, 4);
		matter->ka = vec3(0.1f, 0.1f, 0.1f);
		matter->shininess = 300;

		Material* buraMatter = new Material;
		buraMatter->kd = vec3(0.8f, 0.8f, 0.8f);
		buraMatter->ks = vec3(4, 4, 4);
		buraMatter->ka = vec3(0.1f, 0.1f, 0.1f);
		buraMatter->alwaysShine = true;
		buraMatter->shininess = 300;

		Material* floormatter = new Material;
		floormatter->kd = vec3(0.2f, 0.2f, 0.9f);
		floormatter->ks = vec3(4, 4, 4);
		floormatter->ka = vec3(0.1f, 0.1f, 0.1f);
		floormatter->shininess = 500;

		Geometry* sphere = new Sphere();
		Geometry* cylinder = new Cylinder();
		Geometry* paraboloid = new Paraboloid();
		Geometry* circle = new Circle();

		Object* floor = new Object(shader, floormatter, circle, vec3(0, -3, 0));
		floor->scale = vec3(50,50,50);
		objects.push_back(floor);

		Object* talp = new Object(shader, matter, cylinder, vec3(0, -3, 0));
		talp->scale = vec3(2, 0.5, 2);
		objects.push_back(talp);

		Object* fedo = new Object(shader, matter, circle, vec3(0, -2.5, 0));
		fedo->scale = vec3(2, 0.5, 2);
		objects.push_back(fedo);

		Object* csuklo1 = new Object(shader, matter, sphere, vec3(0, -2.5, 0));
		csuklo1->scale = vec3(0.3f, 0.3f, 0.3f);
		objects.push_back(csuklo1);

		Object* rud1 = new Object(shader, matter, cylinder, vec3(0, -2.5, 0));
		rud1->rotationAxis = vec3(0, 2, 1);
		rud1->scale = vec3(0.2, 3, 0.2);
		objects.push_back(rud1);

		Object* csuklo2 = new Object(shader, matter, sphere, vec3(0, 3, 0));
		csuklo2->rotationAxis = vec3(3, 10, 5);
		csuklo2->scale = vec3(0.3f, 0.3f, 0.3f);
		rud1->setChild(csuklo2);

		Object* rud2 = new Object(shader, matter, cylinder, vec3(0, 0, 0));
		rud2->scale = vec3(0.2, 3, 0.2);
		csuklo2->setChild(rud2);

		Object* csuklo3 = new Object(shader, matter, sphere, vec3(0, 3, 0));
		csuklo3->scale = vec3(0.3f, 0.3f, 0.3f);
		rud2->setChild(csuklo3);


		Object* bura = new Object(shader, buraMatter, paraboloid, vec3(0, 0, 0));
		bura->rotationAxis = vec3(5, -3, 0);
		bura->scale = vec3(1, 0.5f, 1);
		csuklo3->setChild(bura);

		camera.wEye = vec3(0, 2, 12);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		lights.resize(2);
		lights[0].wLightPos = vec4(0, -0.5, 3, 1);
		lights[0].La = vec3(1, 1, 1);
		lights[0].Le = vec3(0.75, 0.75, 0.75);
		lights[0].dir = vec3(0, 1, 0);
		lights[0].fov = 1;

		vec3 fp = bura->refPoint;
		lights[1].wLightPos = vec4(fp.x, fp.y, fp.z, 1);
		lights[1].La = vec3(1, 1, 1);
		lights[1].Le = vec3(0.8, 0.6, 0.1);
		lights[1].dir = vec3(0, 1, 0);
		lights[1].fov = -0.38;

	}

	void Render() {
		RenderState state;
		state.M = state.Minv = mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(0, 0, 0, 1));
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;

		Object* rud1 = objects.back();

		for (Object* obj : objects) obj->Draw(state);

		const vec3 finalPos = rud1->getLastRefPoint();
		vec4 finalPosv4 = vec4(finalPos.x, finalPos.y, finalPos.z, 1);
		lights[1].wLightPos = finalPosv4;
		lights[1].dir = normalize(finalPos);
		rud1->shader->setUniform(lights[1].wLightPos, "lights[1].wLightPos");
	}

	void Animate(float tstart, float tend) {
		Object* rud1 = objects.back();
		rud1->Animate(tstart, tend);
		camera.Animate(tstart - tend);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) { }

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}