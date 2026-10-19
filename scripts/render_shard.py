"""Render a range of frames: Blender Cycles (bpy) -> numpy post (lensing/bloom/HUD) -> PNG.
Run with the python that has `bpy` installed:  python render_shard.py --start 0 --end 120 ..."""
import argparse, math, os, sys, time, tempfile
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import bpy
from mathutils import Matrix, Vector
from config import *
import physics as P
from camrig import rig
import postfx

def mk_node(nt, typ, x=0, y=0, **kw):
    n = nt.nodes.new(typ); n.location = (x, y)
    for k, v in kw.items(): setattr(n, k, v)
    return n

def link(nt, a, ao, b, bi):
    nt.links.new(a.outputs[ao] if isinstance(ao, (int, str)) else ao, b.inputs[bi])

def math_n(nt, op, a=None, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath"); n.operation = op; n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None: continue
        if isinstance(v, bpy.types.NodeSocket): nt.links.new(v, n.inputs[i])
        elif hasattr(v, "outputs"): nt.links.new(v.outputs[0], n.inputs[i])
        else: n.inputs[i].default_value = v
    return n

def value_node(nt, v, label):
    n = nt.nodes.new("ShaderNodeValue"); n.outputs[0].default_value = v; n.label = label; n.name = label
    return n

def new_mat(name):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    return m, nt

def black_material():
    m, nt = new_mat("HorizonBlack")
    e = nt.nodes.new("ShaderNodeEmission"); e.inputs["Color"].default_value = (0, 0, 0, 1); e.inputs["Strength"].default_value = 0
    o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m

def ramp(nt, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    cr = r.color_ramp
    while len(cr.elements) > len(stops): cr.elements.remove(cr.elements[-1])
    while len(cr.elements) < len(stops): cr.elements.new(0.5)
    for el, (p, c) in zip(cr.elements, stops):
        el.position = p; el.color = (*c, 1.0)
    return r

def disk_material(name, r_in, doppler_sign=1.0):
    m, nt = new_mat(name)
    ctrl = dict(rout=value_node(nt, 4.0, "ROUT"), time=value_node(nt, 0.0, "TIME"),
                gain=value_node(nt, 0.5, "GAIN"), spin=value_node(nt, 1.0, "SPINW"))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); link(nt, tc, "Object", sep, 0)
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; link(nt, tc, "Object", ln, 0)
    r = ln.outputs["Value"]
    u = math_n(nt, "DIVIDE", r, ctrl["rout"])
    th = math_n(nt, "ARCTAN2", sep.outputs[1], None); nt.links.new(sep.outputs[0], th.inputs[1])
    rr = math_n(nt, "DIVIDE", r, r_in)
    omega = math_n(nt, "POWER", rr, -1.5)
    omega = math_n(nt, "MULTIPLY", omega, ctrl["spin"])
    dphi = math_n(nt, "MULTIPLY", omega, ctrl["time"])
    phi = math_n(nt, "SUBTRACT", th, dphi)
    cs = math_n(nt, "COSINE", phi); sn = math_n(nt, "SINE", phi)
    k = 2.3
    cx = math_n(nt, "MULTIPLY", cs, k); sy = math_n(nt, "MULTIPLY", sn, k); zz = math_n(nt, "MULTIPLY", rr, 1.6)
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(cx.outputs[0], comb.inputs[0]); nt.links.new(sy.outputs[0], comb.inputs[1]); nt.links.new(zz.outputs[0], comb.inputs[2])
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.noise_dimensions = "3D"
    n1.inputs["Scale"].default_value = 1.0; n1.inputs["Detail"].default_value = 7.0; n1.inputs["Roughness"].default_value = 0.6
    nt.links.new(comb.outputs[0], n1.inputs["Vector"])
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.noise_dimensions = "3D"
    n2.inputs["Scale"].default_value = 3.0; n2.inputs["Detail"].default_value = 4.0
    nt.links.new(comb.outputs[0], n2.inputs["Vector"])
    pat = math_n(nt, "MULTIPLY", n1.outputs["Fac"], 1.6)
    pat = math_n(nt, "ADD", pat, math_n(nt, "MULTIPLY", n2.outputs["Fac"], 0.5)); pat = math_n(nt, "SUBTRACT", pat, 0.45, clamp=True)
    # temperature profile
    temp = math_n(nt, "POWER", rr, -0.75)
    col = ramp(nt, [(0.0, (0.25, 0.03, 0.01)), (0.28, (0.8, 0.18, 0.03)), (0.55, (1.0, 0.55, 0.16)), (0.8, (1.0, 0.82, 0.55)), (1.0, (1.0, 0.95, 0.88))])
    nt.links.new(temp.outputs[0], col.inputs[0])
    bprof = math_n(nt, "POWER", rr, -2.2)
    # doppler beaming
    tan = nt.nodes.new("ShaderNodeCombineXYZ")
    negy = math_n(nt, "MULTIPLY", sep.outputs[1], -1.0)
    nt.links.new(negy.outputs[0], tan.inputs[0]); nt.links.new(sep.outputs[0], tan.inputs[1])
    nrm = nt.nodes.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"; nt.links.new(tan.outputs[0], nrm.inputs[0])
    vt = nt.nodes.new("ShaderNodeVectorTransform"); vt.vector_type = "VECTOR"; vt.convert_from = "OBJECT"; vt.convert_to = "WORLD"
    nt.links.new(nrm.outputs[0], vt.inputs[0])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    nt.links.new(vt.outputs[0], dot.inputs[0]); nt.links.new(geo.outputs["Incoming"], dot.inputs[1])
    dv = math_n(nt, "MULTIPLY", dot, 0.55 * doppler_sign)
    dv = math_n(nt, "ADD", dv, 1.0); dv = math_n(nt, "MAXIMUM", dv, 0.15); dv = math_n(nt, "POWER", dv, 3.0)
    # emission strength
    s1 = math_n(nt, "MULTIPLY", bprof, pat); s2 = math_n(nt, "MULTIPLY", s1, dv); s3 = math_n(nt, "MULTIPLY", s2, ctrl["gain"])
    em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(col.outputs[0], em.inputs["Color"]); nt.links.new(s3.outputs[0], em.inputs["Strength"])
    # alpha
    mo = nt.nodes.new("ShaderNodeMapRange"); mo.clamp = True
    mo.inputs["From Min"].default_value = 0.72; mo.inputs["From Max"].default_value = 1.0; mo.inputs["To Min"].default_value = 1.0; mo.inputs["To Max"].default_value = 0.0
    nt.links.new(u.outputs[0], mo.inputs["Value"])
    mi = nt.nodes.new("ShaderNodeMapRange"); mi.clamp = True
    mi.inputs["From Min"].default_value = r_in * 0.98; mi.inputs["From Max"].default_value = r_in * 1.15
    nt.links.new(r, mi.inputs["Value"])
    al = math_n(nt, "MULTIPLY", mo.outputs[0], mi.outputs[0])
    pa = math_n(nt, "MULTIPLY", pat, 1.2); pa = math_n(nt, "ADD", pa, 0.35, clamp=True)
    al = math_n(nt, "MULTIPLY", al, pa)
    inv = math_n(nt, "SUBTRACT", 1.0, al, clamp=True)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader"); nt.links.new(inv.outputs[0], mix.inputs[0])
    nt.links.new(em.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m, ctrl

def halo_material(name):
    """Camera-facing ring: photon ring + lensed far side of the disk."""
    m, nt = new_mat(name)
    ctrl = dict(rad=value_node(nt, 1.4, "RAD"), gain=value_node(nt, 1.0, "HGAIN"), time=value_node(nt, 0.0, "TIME"))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); link(nt, tc, "Object", sep, 0)
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"; link(nt, tc, "Object", ln, 0)
    rho = math_n(nt, "DIVIDE", ln.outputs["Value"], ctrl["rad"])
    th = math_n(nt, "ARCTAN2", sep.outputs[1], None); nt.links.new(sep.outputs[0], th.inputs[1])
    ph = math_n(nt, "SUBTRACT", th, math_n(nt, "MULTIPLY", ctrl["time"], 0.05))
    cs = math_n(nt, "MULTIPLY", math_n(nt, "COSINE", ph), 1.8); sn = math_n(nt, "MULTIPLY", math_n(nt, "SINE", ph), 1.8)
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(cs.outputs[0], comb.inputs[0]); nt.links.new(sn.outputs[0], comb.inputs[1]); nt.links.new(math_n(nt, "MULTIPLY", rho, 2.0).outputs[0], comb.inputs[2])
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.noise_dimensions = "3D"; nz.inputs["Detail"].default_value = 5.0
    nt.links.new(comb.outputs[0], nz.inputs["Vector"])
    patt = math_n(nt, "ADD", math_n(nt, "MULTIPLY", nz.outputs["Fac"], 1.0), 0.35, clamp=True)
    # photon ring: thin gaussian just outside the shadow; halo: falloff to rho=2.3
    d1 = math_n(nt, "SUBTRACT", rho, 1.035); g1 = math_n(nt, "POWER", math_n(nt, "MULTIPLY", d1, d1), 1.0)
    ring = math_n(nt, "EXPONENT", math_n(nt, "MULTIPLY", g1, -2600.0))
    halo = math_n(nt, "POWER", math_n(nt, "MAXIMUM", math_n(nt, "SUBTRACT", rho, 0.97), 0.001), -2.4)
    fall = nt.nodes.new("ShaderNodeMapRange"); fall.clamp = True
    fall.inputs["From Min"].default_value = 1.1; fall.inputs["From Max"].default_value = 2.4
    fall.inputs["To Min"].default_value = 1.0; fall.inputs["To Max"].default_value = 0.0
    nt.links.new(rho.outputs[0], fall.inputs["Value"])
    halo = math_n(nt, "MULTIPLY", math_n(nt, "MULTIPLY", halo, fall.outputs[0]), math_n(nt, "MULTIPLY", patt, ctrl["gain"]))
    halo = math_n(nt, "MULTIPLY", halo, 0.45)
    inner = nt.nodes.new("ShaderNodeMapRange"); inner.clamp = True
    inner.inputs["From Min"].default_value = 0.985; inner.inputs["From Max"].default_value = 1.0
    nt.links.new(rho.outputs[0], inner.inputs["Value"])
    ringp = math_n(nt, "MULTIPLY", ring, 0.9)
    strength = math_n(nt, "MULTIPLY", math_n(nt, "ADD", halo, ringp), inner.outputs[0])
    col = ramp(nt, [(0.0, (0.5, 0.1, 0.02)), (0.35, (1.0, 0.45, 0.1)), (0.7, (1.0, 0.8, 0.5)), (1.0, (1.0, 0.95, 0.85))])
    cin = nt.nodes.new("ShaderNodeMapRange"); cin.clamp = True
    cin.inputs["From Min"].default_value = 2.4; cin.inputs["From Max"].default_value = 1.0
    nt.links.new(rho.outputs[0], cin.inputs["Value"])
    nt.links.new(cin.outputs[0], col.inputs[0])
    em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(col.outputs[0], em.inputs["Color"]); nt.links.new(strength.outputs[0], em.inputs["Strength"])
    al = math_n(nt, "MULTIPLY", math_n(nt, "MINIMUM", math_n(nt, "ADD", halo, ringp), 1.0), inner.outputs[0])
    inv = math_n(nt, "SUBTRACT", 1.0, al, clamp=True)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader"); nt.links.new(inv.outputs[0], mix.inputs[0])
    nt.links.new(em.outputs[0], mix.inputs[1]); nt.links.new(tr.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m, ctrl

def annulus(name, r_in, r_out, seg=192):
    me = bpy.data.meshes.new(name)
    verts, faces = [], []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        verts += [(r_in * math.cos(a), r_in * math.sin(a), 0), (r_out * math.cos(a), r_out * math.sin(a), 0)]
    for i in range(seg):
        j = (i + 1) % seg
        faces.append((2 * i, 2 * i + 1, 2 * j + 1, 2 * j))
    me.from_pydata(verts, [], faces); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    return ob

def sphere(name, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
    ob = bpy.context.active_object; ob.name = name
    bpy.ops.object.shade_smooth()
    ob.data.materials.append(mat)
    return ob

class Scene:
    def __init__(self, W, H, samples, doppler_sign):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene; self.sc = sc
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.02
        sc.cycles.use_denoising = True
        try: sc.cycles.denoiser = "OPENIMAGEDENOISE"
        except Exception: pass
        sc.cycles.max_bounces = 2; sc.cycles.transparent_max_bounces = 16
        sc.render.film_transparent = True
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
        sc.render.image_settings.file_format = "OPEN_EXR"
        sc.render.image_settings.color_depth = "16"
        sc.render.image_settings.color_mode = "RGBA"
        try: sc.render.image_settings.exr_codec = "ZIP"
        except Exception: pass
        sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
        sc.render.use_persistent_data = True
        w = bpy.data.worlds.new("W"); sc.world = w
        cam = bpy.data.cameras.new("Cam"); cam.sensor_fit = "HORIZONTAL"; cam.sensor_width = 36.0
        cam.clip_start, cam.clip_end = 0.1, 2000
        self.cam_ob = bpy.data.objects.new("Cam", cam); sc.collection.objects.link(self.cam_ob); sc.camera = self.cam_ob
        blk = black_material()
        self.bh = [sphere("BH1", blk), sphere("BH2", blk)]
        self.disk_ctrl, self.disks, self.halo_ctrl, self.halos = [], [], [], []
        for i, (rs, nm) in enumerate(((2 * m1, "D1"), (2 * m2, "D2"), (2 * mf, "DF"))):
            r_in = SHADOW * rs * 1.25
            mat, ctrl = disk_material(nm, r_in, doppler_sign)
            ob = annulus(nm, r_in, SHADOW * rs * 6.0)
            ob.data.materials.append(mat)
            self.disks.append(ob); self.disk_ctrl.append(ctrl)
        for i in range(2):
            mat, ctrl = halo_material(f"H{i}")
            ob = annulus(f"H{i}", 0.2, 3.0, seg=96); ob.data.materials.append(mat)
            self.halos.append(ob); self.halo_ctrl.append(ctrl)
        self.W, self.H = W, H

    def set_state(self, st, cam):
        sc = self.sc
        # camera
        M = Matrix([[cam["R"][0], cam["U"][0], cam["B"][0], cam["pos"][0]],
                    [cam["R"][1], cam["U"][1], cam["B"][1], cam["pos"][1]],
                    [cam["R"][2], cam["U"][2], cam["B"][2], cam["pos"][2]], [0, 0, 0, 1]])
        self.cam_ob.matrix_world = M
        self.cam_ob.data.lens = float(cam["lens"])
        merged = st["tau"] <= 0 or st["k"] > 0.95
        view_up = abs(float(np.dot(cam["B"], [0, 0, 1])))
        edge = 1.0 - view_up
        holes = [(st["p1"], st["r1"], 0), (st["p2"], st["r2"], 1)]
        for ob, (p, r, i) in zip(self.bh, holes):
            if merged and i == 1:
                ob.hide_render = True; continue
            ob.hide_render = False
            pos = np.zeros(3) if merged else p
            ob.location = Vector(pos.tolist())
            sx = sy = sz = r
            if st["tau"] <= 0:
                e = st["wob"]; sx, sy, sz = r * (1 + e), r * (1 - e), r
                ob.rotation_euler = (0, 0, st["phi"] * 0.5)
            else:
                ob.rotation_euler = (0, 0, 0)
            ob.scale = (sx, sy, sz)
        # disks: D1,D2 for holes, DF for remnant
        spec = [(st["rout1"], st["p1"], 0, 0), (st["rout2"], st["p2"], 1, 1), (st["rout_f"], np.zeros(3), 2, 2)]
        for ob, ctrl, (rout, p, i, ci) in zip(self.disks, self.disk_ctrl, spec):
            vis = rout > 0.4 and not (merged and i < 2)
            ob.hide_render = not vis
            if not vis: continue
            ob.location = Vector(p.tolist())
            ctrl["rout"].outputs[0].default_value = rout
            ctrl["time"].outputs[0].default_value = st["t_story"]
            ctrl["spin"].outputs[0].default_value = 0.10 * (m1 if i == 0 else m2 if i == 1 else mf) ** 0.5
            ctrl["gain"].outputs[0].default_value = 0.5
        # halos
        for hob, ctrl, (p, r, i) in zip(self.halos, self.halo_ctrl, holes):
            if merged and i == 1:
                hob.hide_render = True; continue
            hob.hide_render = False
            pos = np.zeros(3) if merged else p
            n = cam["B"]
            Mh = Matrix([[cam["R"][0], cam["U"][0], n[0], pos[0]], [cam["R"][1], cam["U"][1], n[1], pos[1]],
                         [cam["R"][2], cam["U"][2], n[2], pos[2]], [0, 0, 0, 1]])
            hob.matrix_world = Mh
            ctrl["rad"].outputs[0].default_value = r
            ctrl["time"].outputs[0].default_value = st["t_story"]
            ctrl["gain"].outputs[0].default_value = 0.04 + 0.22 * edge ** 0.7
            hob.scale = (1, 1, 1)
        # scale halo meshes with hole radius (mesh built for r in [0.2,3]; shader uses RAD in object space)
        for hob, (p, r, i) in zip(self.halos, holes):
            hob.scale = (r / 1.4, r / 1.4, 1)
            # shader RAD is in object space: object-space radius of the shadow edge is 1.4 (unscaled)
        for ctrl in self.halo_ctrl:
            ctrl["rad"].outputs[0].default_value = 1.4

    def render_exr(self, path):
        self.sc.render.filepath = path
        bpy.ops.render.render(write_still=True)

def read_exr(path):
    """Read raw (as stored, premultiplied) RGBA float EXR via Blender's own loader."""
    im = bpy.data.images.load(path, check_existing=False)
    try:
        im.alpha_mode = "CHANNEL_PACKED"
    except Exception:
        pass
    w, h = im.size
    buf = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(buf)
    bpy.data.images.remove(im)
    return buf.reshape(h, w, 4)[::-1].copy()      # Blender rows are bottom-up

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--frames", type=str, default="")   # comma list overrides range
    ap.add_argument("--res", type=int, default=1080); ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--samples", type=int, default=64); ap.add_argument("--assets", default="../assets")
    ap.add_argument("--out", default="frames"); ap.add_argument("--doppler-sign", type=float, default=1.0)
    ap.add_argument("--no-hud", action="store_true")
    a = ap.parse_args()
    H = a.res; W = H * 16 // 9
    os.makedirs(a.out, exist_ok=True)
    sky = postfx.Sky(a.assets)
    sc = Scene(W, H, a.samples, a.doppler_sign)
    idx = [int(x) for x in a.frames.split(",")] if a.frames else list(range(a.start, a.end, a.step))
    tmp = tempfile.mkdtemp(); t_all = time.time()
    for n, fr in enumerate(idx):
        t0 = time.time()
        t = fr / a.fps
        shot = P.shot_at(t)
        tau = P.tau_at(t, shot); st = P.state(tau)
        cam = rig(t, st, shot)
        sc.set_state(st, cam)
        exr = os.path.join(tmp, "f.exr")
        sc.render_exr(exr)
        rgba = read_exr(exr)
        t1 = time.time()
        bg = postfx.lensed_sky(sky, cam, st, W, H)
        lin = postfx.composite(rgba, bg)
        fade = min(1.0, t / 1.5, (DURATION_S - t) / 2.5)
        img = postfx.finish(lin, W, H, fade=max(fade, 0.0), seed=fr)
        if not a.no_hud:
            img = postfx.hud_overlay(img, t, st, shot, sky, a.fps)
        cv2.imwrite(os.path.join(a.out, f"f_{fr:06d}.png"), img[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f"frame {fr} ({n+1}/{len(idx)}) shot={shot['id']} cycles={t1-t0:.1f}s post={time.time()-t1:.1f}s total={time.time()-t_all:.0f}s", flush=True)

if __name__ == "__main__":
    main()
