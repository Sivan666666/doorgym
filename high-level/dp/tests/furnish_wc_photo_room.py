"""Photo-guided office reconstruction. Dimensions are estimates, not a scan.

Coordinate system: door centre x=0, door plane y=0, entrance y<0,
office y>0, +Z up. Never modify the existing door or robot assets.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import bpy
import numpy as np
from mathutils import Vector
from augmentation import material_manifest


def furnish_room(scene, data, convert, pose, source, out, settings=None):
    settings = settings or {}
    room = settings.get('photo_room', {})
    light_multiplier = float(room.get('light_multiplier', 1.0))
    assert math.isfinite(light_multiplier) and light_multiplier >= 0
    rng = random.Random(20261008)
    created = []
    solid = []
    light_report = []
    emission_report = []
    collection = bpy.data.collections.new('Photo reconstruction / 20261008')
    scene.collection.children.link(collection)

    def register(obj, category='prop'):
        for col in list(obj.users_collection):col.objects.unlink(obj)
        collection.objects.link(obj)
        obj['photo_category'] = category
        created.append(obj)
        if obj.type == 'MESH' and category not in ('floor','decoration','exterior'):
            solid.append(obj)
        return obj

    def material(name, color, roughness=.65, metallic=0., emission=0.):
        m = bpy.data.materials.new('Photo / '+name)
        m.use_nodes = True
        m.diffuse_color = (*color,1)
        bs = m.node_tree.nodes.get('Principled BSDF')
        bs.inputs['Base Color'].default_value = (*color,1)
        bs.inputs['Roughness'].default_value = roughness
        bs.inputs['Metallic'].default_value = metallic
        if emission:
            bs.inputs['Emission Color'].default_value = (*color,1)
            bs.inputs['Emission Strength'].default_value = emission * light_multiplier
            emission_report.append(dict(material=m.name,base_strength=emission,
                strength=emission * light_multiplier))
        return m

    def noise_material(name, low, high, scale, stretch, rough=.8, bump=.002):
        m = material(name, low, rough)
        n,l = m.node_tree.nodes,m.node_tree.links
        coord=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY'
        mapping.inputs[1].default_value=stretch
        l.new(coord.outputs['Object'],mapping.inputs[0])
        noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=scale
        noise.inputs['Detail'].default_value=3.5;noise.inputs['Roughness'].default_value=.72
        l.new(mapping.outputs[0],noise.inputs['Vector'])
        ramp=n.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position=.18;ramp.color_ramp.elements[0].color=(*low,1)
        ramp.color_ramp.elements[1].position=.82;ramp.color_ramp.elements[1].color=(*high,1)
        l.new(noise.outputs['Fac'],ramp.inputs[0])
        bs=n.get('Principled BSDF');l.new(ramp.outputs[0],bs.inputs['Base Color'])
        b=n.new('ShaderNodeBump');b.inputs['Strength'].default_value=.22;b.inputs['Distance'].default_value=bump
        l.new(noise.outputs['Fac'],b.inputs['Height']);l.new(b.outputs[0],bs.inputs['Normal'])
        return m

    plaster=material('warm off white wall',(.72,.715,.68),.9)
    white=material('cabinet melamine',(.78,.79,.77),.48)
    black=material('black skirting rubber',(.013,.014,.016),.58)
    plastic=material('black ABS',(.012,.015,.019),.36)
    screen=material('unlit monitor glass',(.012,.016,.021),.22,.10)
    steel=material('silver aluminium',(.38,.40,.42),.32,.7)
    darksteel=material('dark powder coat',(.025,.030,.035),.48,.18)
    floor=noise_material('grey streaked PVC floor',(.25,.25,.225),(.41,.405,.365),5,(3,40,2),.86,.003)
    oak=noise_material('light oak desktops',(.39,.23,.10),(.65,.44,.23),4,(2,60,3),.52,.001)
    cardboard=noise_material('brown corrugated cardboard',(.31,.19,.085),(.48,.32,.16),32,(1,1,1),.92,.0008)
    tape=material('packing tape',(.48,.33,.16),.3)
    fabric=noise_material('dark woven fabric',(.009,.012,.015),(.045,.050,.06),140,(1,1,1),.94,.001)
    roof=material('dark office ceiling',(.027,.025,.025),.95)
    tile=noise_material('speckled acoustic ceiling tile',(.63,.64,.60),(.82,.82,.77),85,(1,1,1),.95,.002)
    blue=material('muted turquoise objects',(.035,.36,.35),.45)
    purple=material('purple bag',(.14,.025,.29),.9)
    red=material('red folder',(.48,.015,.018),.66)
    paper=material('paper',(.83,.82,.78),.87)
    green=material('emergency green',(.012,.19,.115),.7)

    def box(name,center,size,mat,bevel=0.,category='prop',rotation=0.):
        bpy.ops.mesh.primitive_cube_add(size=1,location=center)
        o=bpy.context.object;o.name='Photo / '+name;o.scale=size
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        o.rotation_euler.z=rotation
        o.data.materials.append(mat)
        if bevel:
            b=o.modifiers.new('edge bevel','BEVEL');b.width=bevel;b.segments=3
        return register(o,category)

    def rod(name,a,b,r,mat,category='prop',vertices=12):
        a,b=Vector(a),Vector(b);delta=b-a
        bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=delta.length,location=(a+b)/2)
        o=bpy.context.object;o.name='Photo / '+name
        o.rotation_euler=delta.to_track_quat('Z','Y').to_euler();o.data.materials.append(mat)
        for poly in o.data.polygons:poly.use_smooth=True
        return register(o,category)

    def cable(name,points,mat=black,r=.004):
        c=bpy.data.curves.new('Photo / '+name,'CURVE');c.dimensions='3D';c.bevel_depth=r;c.bevel_resolution=2
        sp=c.splines.new('POLY');sp.points.add(len(points)-1)
        for p,co in zip(sp.points,points):p.co=(*co,1)
        o=bpy.data.objects.new(c.name,c);collection.objects.link(o);o.data.materials.append(mat)
        o['photo_category']='decoration';created.append(o)
        return o

    def text(name,body,loc,size,mat,rotation=(math.pi/2,0,0)):
        c=bpy.data.curves.new('Photo / '+name,'FONT');c.body=body;c.size=size;c.extrude=.0001
        o=bpy.data.objects.new(c.name,c);collection.objects.link(o);o.location=loc;o.rotation_euler=rotation
        c.materials.append(mat);created.append(o);o['photo_category']='decoration'
        return o

    # Hide only the previous environment. Original door/robot assignments stay intact.
    protected=set()
    for root_name in ('frame','door_leaf','lever_handle','robot/world'):
        root=bpy.data.objects.get(root_name)
        if root:
            protected.add(root);protected.update(root.children_recursive)
    protected.update(o for o in bpy.data.objects if o.name.startswith('robot/'))
    def asset_signature():
        meshes={}
        names=set()
        for obj in sorted(protected,key=lambda o:o.name):
            if obj.type!='MESH':continue
            mats=[m.name if m else None for m in obj.data.materials]
            names.update(n for n in mats if n)
            meshes[obj.name]=dict(vertices=[list(v.co) for v in obj.data.vertices],
                polygons=[list(p.vertices) for p in obj.data.polygons],
                material_indices=[p.material_index for p in obj.data.polygons],materials=mats,
                hide_render=obj.hide_render)
        manifest=material_manifest(bpy)
        value=dict(meshes=meshes,materials={n:manifest[n] for n in sorted(names)})
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    asset_before=asset_signature()
    for o in list(bpy.data.objects):
        if o.type=='LIGHT':o.data.energy=0
        if o.type=='MESH' and o not in protected and not o.name.startswith('robot/'):
            o.hide_render=True
    world=scene.world
    if world and world.use_nodes:
        for n in world.node_tree.nodes:
            if n.type=='BACKGROUND':n.inputs['Strength'].default_value=0.

    left,right=-1.65,2.8
    back=4.8
    entry_left,entry_right=-.78,2.8
    front=-3.8
    h_entry,h_office=2.78,3.05
    door_bounds=[o.matrix_world@Vector(c) for o in bpy.data.objects['frame'].children if o.type=='MESH' for c in o.bound_box]
    dlo,dhi,dt=min(v.x for v in door_bounds),max(v.x for v in door_bounds),max(v.z for v in door_bounds)
    # New walls closely follow photographed corners, not the old 8 m wide room.
    box('entrance left wall',(entry_left-.06,front/2,h_entry/2),(.12,-front,h_entry),plaster,category='wall')
    box('entrance right wall',(entry_right+.06,front/2,h_entry/2),(.12,-front,h_entry),plaster,category='wall')
    box('entrance rear wall',((entry_left+entry_right)/2,front-.06,h_entry/2),(entry_right-entry_left,.12,h_entry),plaster,category='wall')
    box('door surround left',((left+dlo)/2,.105,h_entry/2),(dlo-left,.16,h_entry),plaster,category='wall')
    box('door surround right',((dhi+right)/2,.105,h_entry/2),(right-dhi,.16,h_entry),plaster,category='wall')
    box('door lintel',((dlo+dhi)/2,.105,(dt+h_entry)/2),(dhi-dlo,.16,h_entry-dt),plaster,category='wall')
    box('office left wall',(left-.06,back/2,h_office/2),(.12,back,h_office),plaster,category='wall')
    box('office right wall',(right+.06,back/2,h_office/2),(.12,back,h_office),plaster,category='wall')
    # Back wall split around the large night window.
    wx0,wx1,wz0,wz1=-.85,1.70,1.0,2.58
    box('back wall under window',((left+right)/2,back+.06,wz0/2),(right-left,.12,wz0),plaster,category='wall')
    box('back wall over window',((left+right)/2,back+.06,(wz1+h_office)/2),(right-left,.12,h_office-wz1),plaster,category='wall')
    for name,x0,x1 in [('left',left,wx0),('right',wx1,right)]:
        box('back wall '+name,((x0+x1)/2,back+.06,(wz0+wz1)/2),(x1-x0,.12,wz1-wz0),plaster,category='wall')
    box('entrance PVC floor',((entry_left+entry_right)/2,front/2,-.085),(entry_right-entry_left,-front,.06),floor,category='floor')
    box('office PVC floor',((left+right)/2,back/2,-.085),(right-left,back,.06),floor,category='floor')
    seam=material('PVC seams',(.22,.23,.23),.85)
    for y in np.arange(front,back,.9):
        x0=entry_left if y<0 else left
        box('subtle floor seam',((x0+right)/2,float(y),-.054),(right-x0,.0012,.0005),seam,category='decoration')
    for name,c,s in [
        ('entry left',(entry_left+.009,front/2,.018),(.018,-front,.145)),
        ('entry right',(entry_right-.009,front/2,.018),(.018,-front,.145)),
        ('front surround L',((dlo+entry_left)/2,.012,.018),(dlo-entry_left,.018,.145)),
        ('front surround R',((dhi+right)/2,.012,.018),(right-dhi,.018,.145)),
        ('office left',(left+.009,back/2,.018),(.018,back,.145)),
        ('office right',(right-.009,back/2,.018),(.018,back,.145)),
        ('office back',((left+right)/2,back-.012,.018),(right-left,.024,.145))]:
        box('black skirting '+name,c,s,black,category='wall')

    # Door-side access switch, blank gang, emergency release, thermostat, and door stop.
    for i,mat in enumerate((white,white,green,white)):
        x=dhi+.12+i*.085
        box('wall control '+str(i),(x,-.002,1.36),(.078,.026,.084),mat,.003,'decoration')
        if i==3:
            box('thermostat LCD',(x,-.017,1.374),(.047,.003,.026),material('LCD',(.13,.16,.14),.45),category='decoration')
            for dx in (-.02,0,.02):box('thermostat key',(x+dx,-.019,1.339),(.008,.003,.004),darksteel,category='decoration')
        if i==2:
            for z in (1.385,1.372,1.354):box('release small lettering',(x,-.017,z),(.050,.002,.002),white,category='decoration')
    rod('wall doorstop',(entry_left,-.55,.14),(entry_left+.085,-.55,.14),.017,darksteel,'decoration')
    box('magnetic lock above opening',(.31,.125,2.22),(.26,.052,.043),steel,.003,'decoration')
    box('magnet inset',(.31,.095,2.22),(.18,.004,.026),darksteel,category='decoration')
    # Loose white cable in the same entrance right-floor region as photos 1/2.
    for j in range(9):
        pts=[]
        for k in range(90):
            t=k/89*math.tau*1.7
            pts.append((1.20+.20*math.cos(t+j*.8)+j*.006,-.78+.14*math.sin(t*1.07+j*.3),-.043+j*.001))
        cable('entrance coiled white cable',pts,paper,.0015)

    # Entrance suspended 600 mm ceiling grid, one bright square and a darker tile.
    cell=.6
    for ix in range(6):
        x=-.48+ix*cell
        for iy in range(7):
            y=-.32-iy*cell
            if y<front or x>entry_right:continue
            is_panel=(ix==1 and iy==1)
            is_dark=(ix==0 and iy==0)
            mat=material('dark replacement ceiling tile',(.16,.17,.17),.75) if is_dark else tile
            if is_panel:mat=material('entrance panel diffuser',(1.,.96,.86),.4,emission=1.5)
            box('entrance ceiling '+('LED' if is_panel else 'tile'),(x,y,h_entry+.018),(.587,.587,.035),mat,category='ceiling')
    for x in np.arange(entry_left,right+.1,.6):
        box('ceiling T rail', (float(x),front/2,h_entry-.004),(.012,-front,.012),steel,category='ceiling')
    for y in np.arange(-.02,front-.1,-.6):
        box('ceiling cross T rail',((entry_left+right)/2,float(y),h_entry-.004),(right-entry_left,.012,.012),steel,category='ceiling')
    # Enclose thin clipped edges without creating visual daylight leaks.
    box('ceiling backing',((entry_left+right)/2,front/2,h_entry+.07),(right-entry_left,-front,.04),tile,category='ceiling')

    # Office dark central ceiling with white perimeter dropped soffit.
    box('office dark ceiling',((left+right)/2,back/2,h_office),(right-left,back,.10),roof,category='ceiling')
    for name,c,s in [
        ('left',(left+.21,back/2,2.80),(.42,back,.18)),('right',(right-.21,back/2,2.80),(.42,back,.18)),
        ('front',((left+right)/2,.20,2.80),(right-left,.40,.18)),('back',((left+right)/2,back-.20,2.80),(right-left,.40,.18))]:
        box('office white soffit '+name,c,s,plaster,category='ceiling')
    for y in (1.05,2.85,4.25):
        rod('recessed unlit downlight',(left+.2,y,2.704),(left+.2,y,2.711),.045,white,'decoration',24)
    rod('ceiling security camera mount',(left+.18,.54,2.68),(left+.18,.54,2.71),.048,white,'decoration',24)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.035,location=(left+.18,.54,2.668))
    bpy.context.object.data.materials.append(plastic);register(bpy.context.object,'decoration')
    # Long cool-white pendant matches photos 5/6, over the right workbench.
    box('linear pendant black body',(1.62,2.62,2.58),(.125,2.0,.055),darksteel,.008,'ceiling')
    led=material('office strip diffuser',(.80,.89,1.),.35,emission=4.)
    box('linear pendant luminous underside',(1.62,2.62,2.548),(.095,1.96,.009),led,category='ceiling')
    for y in (1.90,3.35):rod('pendant suspension',(1.62,y,2.62),(1.62,y,3.0),.002,darksteel,'decoration')

    def area(name,loc,energy,color,size,shape='DISK',size_y=None):
        base_energy=energy;energy=energy * light_multiplier
        l=bpy.data.lights.new('Photo / '+name,'AREA');l.energy=energy;l.color=color;l.shape=shape;l.size=size
        if size_y is not None:l.size_y=size_y
        o=bpy.data.objects.new(l.name,l);collection.objects.link(o);o.location=loc
        o.rotation_euler=(0,0,0)
        light_report.append(dict(name=o.name,type='AREA',position=list(loc),energy=energy,base_energy=base_energy,
            multiplier=light_multiplier,color=list(color),size=size,size_y=size_y))
        return o
    area('entrance square panel',(.12,-.92,h_entry-.014),room.get('entrance_panel_energy',65),(1.,.96,.89),.57,'SQUARE')
    area('office linear pendant',(1.62,2.62,2.537),room.get('office_strip_energy',135),(.80,.90,1.),.09,'RECTANGLE',1.94)

    # Large night window with genuine reflective glass and a sparse exterior facade.
    windowframe=material('window powder coated frame',(.15,.17,.18),.42,.2)
    glass=material('night glazing',(.98,.99,1.),.004)
    bs=glass.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Transmission Weight'].default_value=1.;bs.inputs['IOR'].default_value=1.46
    for x in (wx0,wx1,-.04):box('window vertical mullion',(x,back-.018,(wz0+wz1)/2),(.045,.075,wz1-wz0),windowframe,category='wall')
    for z in (wz0,wz1):box('window horizontal frame',((wx0+wx1)/2,back-.018,z),(wx1-wx0,.075,.045),windowframe,category='wall')
    box('night window glass',((wx0+wx1)/2,back+.009,(wz0+wz1)/2),(wx1-wx0-.03,.012,wz1-wz0-.02),glass,category='wall')
    exterior=material('night building facade',(.006,.009,.015),.95)
    box('night exterior backdrop',(.3,back+2.0,3.7),(12,.04,10),exterior,category='exterior')
    litwindow=material('distant lit windows',(.68,.67,.49),.7,emission=4.)
    for xi in range(15):
        for zi in range(14):
            if rng.random()<.40:
                box('distant office window',(-4.2+xi*.60,back+1.97,.2+zi*.47),(.047,.01,.12),litwindow,category='exterior')
    amber=material('distant amber globes',(1.,.42,.10),.6,emission=1.)
    for x in (-.5,.22,1.15):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=6,radius=.035,location=(x,back+1.92,1.1))
        bpy.context.object.data.materials.append(amber);register(bpy.context.object,'exterior')
    # Zebra blind at the top of the large window and a secondary blind on the right wall.
    blind=material('dark zebra blind',(.03,.032,.035),.85)
    for z in np.arange(2.38,2.59,.10):box('back zebra blind stripe',((wx0+wx1)/2,back-.07,float(z)),(wx1-wx0,.015,.062),blind,category='decoration')
    for z in np.arange(1.60,2.60,.075):box('right roller blind',(right-.016,4.05,float(z)),(.02,1.0,.043),tile,category='decoration')

    # Back bank of white cupboards and counter: printer left, display centrally.
    for j in range(6):
        x=-1.35+j*.62
        box('back cupboard '+str(j),(x,4.45,.40),(.608,.60,.86),white,.007)
        box('back cabinet front '+str(j),(x,4.135,.41),(.599,.025,.82),white,.003)
        rod('cabinet slim pull',(x+.23,4.117,.52),(x+.23,4.117,.66),.0035,steel,'decoration')
    box('back continuous countertop',(.20,4.44,.85),(3.76,.66,.04),white,.006)

    def monitor(name,x,y,z,width=.52,yaw=0):
        start=len(created)
        box(name+' screen',(0,0,z+.22),(width,.043,.33),plastic,.009)
        box(name+' display',(0,-.024,z+.225),(width-.027,.003,.301),screen,.003)
        rod(name+' support',(0,.015,z+.03),(0,.015,z+.10),.023,plastic)
        box(name+' base',(0,-.015,z+.014),(.25,.18,.025),darksteel,.015)
        for o in created[start:]:
            vx,vy=o.location.x,o.location.y
            o.location.x=x+math.cos(yaw)*vx-math.sin(yaw)*vy
            o.location.y=y+math.sin(yaw)*vx+math.cos(yaw)*vy
            o.rotation_euler.z+=yaw
    monitor('back monitor',-.23,4.28,.875,.53,.035)
    box('printer main',(-1.20,4.37,1.04),(.45,.38,.34),white,.012)
    box('printer black top',(-1.20,4.37,1.218),(.45,.37,.027),plastic,.004)
    box('printer scanner lid',(-1.20,4.405,1.255),(.41,.30,.04),white,.006)
    box('printer paper slot',(-1.20,4.172,1.04),(.30,.012,.026),plastic,category='decoration')
    for i in range(4):box('printer loose papers',(-1.02,4.1,.883+i*.005),(.23,.19,.003),paper,rotation=-.06)
    for j in range(3):box('counter stacked document trays',(.52,4.30,.90+j*.07),(.50,.30,.027),plastic,.004)
    for j,mat in enumerate((blue,red,oak)):
        box('counter books',(.51,4.27,.93+j*.04),(.39,.23,.025),mat,.002)
    for j in range(5):
        x=.92+j*.075
        box('upright binders',(x,4.49,1.03),(.052,.20,.32),red if j==0 else white,.003)
    # Wires dangling off the back cabinet edge, and the MSI-sized box on floor.
    for j in range(4):cable('counter dangling lead',[(-.43+j*.04,4.17,.9),(-.36+j*.03,4.08,.74),(-.38+j*.05,4.0,.45),(-.22+j*.025,4.07,.49),(-.20,4.2,.85)],black,.002)

    def carton(name,c,size,opened=False):
        x,y,z=c;w,d,h=size
        box(name+' bottom',(x,y,z+.008),(w,d,.016),cardboard)
        for dx in (-w/2,w/2):box(name+' side',(x+dx,y,z+h/2),(.012,d,h),cardboard,.002)
        for dy in (-d/2,d/2):box(name+' end',(x,y+dy,z+h/2),(w,.012,h),cardboard,.002)
        if opened:
            for side in (-1,1):
                o=box(name+' open flap',(x+side*w*.64,y,z+h+.055),(w*.45,d,.009),cardboard)
                o.rotation_euler.y=side*math.radians(-38)
            box(name+' visible packed paper',(x,y,z+h*.70),(w*.8,d*.8,.02),paper)
        else:
            box(name+' lid',(x,y,z+h),(w,d,.015),cardboard)
            box(name+' tape',(x,y,z+h+.009),(.044,d,.003),tape,category='decoration')
        for k in range(3):box(name+' label bar',(x-w*.3+k*.035,y-d/2-.007,z+h*.38),(.017,.002,.018),plastic,category='decoration')

    carton('back floor computer box',(-.12,3.86,-.05),(.65,.26,.37))
    text('MSI box mark','MSI',(-.38,3.72,.08),.10,darksteel)
    box('small equipment case',(.31,3.9,.12),(.11,.23,.34),plastic,.02)
    # Left pile directly below the red notice board, not a shelf.
    box('large black equipment case',(-1.26,1.65,.47),(.62,.62,1.04),plastic,.02)
    for z in (.15,.78):box('case silver latch',(-1.26,1.325,z),(.15,.02,.023),steel,category='decoration')
    text('case brand','Unilumin',(-1.45,1.318,.32),.057,white)
    carton('left tall open carton',(-1.19,2.62,-.05),(.55,.55,.80),True)
    carton('left rear packing carton',(-1.14,3.30,-.05),(.68,.47,.47))
    carton('left carton on case',(-1.28,1.74,1.02),(.57,.48,.38),True)
    carton('low left box',(-1.30,.72,-.05),(.51,.46,.29))
    box('purple folded cloth',(-1.2,1.65,1.14),(.37,.31,.16),purple,.07)
    for x,y,z in [(-1.30,2.04,1.12),(-1.37,2.73,.84)]:
        box('old desktop chassis',(x,y,z),(.25,.42,.16),darksteel,.008)
        for k in range(7):box('case vent',(x-.09+k*.025,y-.218,z),(.008,.003,.065),black,category='decoration')

    # Office chairs: curved padded mesh-like backs, armrests and five caster legs.
    def chair(name,x,y,yaw):
        start=len(created)
        box(name+' seat',(0,0,.47),(.47,.45,.095),fabric,.06)
        # Shaped back surface, more faithful than flat cuboids.
        verts=[];faces=[];rows,cols=9,9
        for iz in range(rows):
            t=iz/(rows-1);z=.54+t*.66;width=.21+.025*math.sin(t*math.pi)
            for ix in range(cols):
                u=ix/(cols-1)*2-1
                verts.append((u*width,.19+.065*u*u+.065*t*t,z))
        for iz in range(rows-1):
            for ix in range(cols-1):
                n=iz*cols+ix;faces.append((n,n+1,n+1+cols,n+cols))
        mesh=bpy.data.meshes.new(name+' curved back');mesh.from_pydata(verts,[],faces);mesh.materials.append(fabric)
        o=bpy.data.objects.new('Photo / '+name+' curved back',mesh);collection.objects.link(o);created.append(o);solid.append(o)
        o['photo_category']='prop'
        so=o.modifiers.new('back thickness','SOLIDIFY');so.thickness=.038
        su=o.modifiers.new('smooth back','SUBSURF');su.levels=2;su.render_levels=2
        for p in mesh.polygons:p.use_smooth=True
        rod(name+' gas lift',(0,0,.12),(0,0,.43),.032,darksteel)
        for i in range(5):
            angle=i*math.tau/5
            pt=(.29*math.cos(angle),.29*math.sin(angle),.055)
            rod(name+' star leg',(0,0,.13),pt,.022,plastic)
            rod(name+' caster',(pt[0]-.035,pt[1],.018),(pt[0]+.035,pt[1],.018),.04,plastic,vertices=16)
        for side in (-1,1):
            rod(name+' arm support',(side*.24,.08,.46),(side*.26,.08,.68),.015,plastic)
            box(name+' arm pad',(side*.26,-.015,.70),(.065,.27,.043),plastic,.018)
        for obj in created[start:]:
            vx,vy=obj.location.x,obj.location.y
            obj.location.x=x+math.cos(yaw)*vx-math.sin(yaw)*vy
            obj.location.y=y+math.sin(yaw)*vx+math.cos(yaw)*vy
            obj.rotation_euler.z+=yaw
    chair('left stored chair',-.99,1.94,-.30)
    # Backpacks leaning at its feet and against the seat.
    bag=box('left floor backpack',(-.98,1.51,.19),(.30,.20,.47),fabric,.07,rotation=-.25)
    bag.rotation_euler.x=.30
    cable('backpack shoulder strap',[(-1.1,1.40,.19),(-1.14,1.38,.46),(-1.0,1.38,.51),(-.90,1.39,.18)],black,.012)

    # Right two-sided wood bench, black monitors, desktop clutter and RGB tower.
    box('right long desk',(1.93,2.85,.74),(1.52,2.62,.05),oak,.012)
    for x in (1.23,2.63):
        for y in (1.61,4.09):box('white desk leg',(x,y,.33),(.047,.047,.76),white,.006)
    for j,y in enumerate((2.18,3.13,3.95)):
        chair('right aisle chair '+str(j),.90,y,math.pi/2+(-.30,.30,.03)[j])
        monitor('desk monitor '+str(j),1.75,y,.77,.49,-math.pi/2)
        box('desk keyboard',(1.37,y-.03,.79),(.17,.42,.025),white,.006)
        for row in range(4):
            for col in range(12):box('keyboard key',(1.30+row*.035,y-.21+col*.032,.806),(.025,.025,.006),white,.002,'decoration')
        box('mouse',(1.35,y+.26,.802),(.065,.10,.037),plastic,.025)
    for j,y in enumerate((2.0,3.15)):
        monitor('opposite workstation monitor '+str(j),2.24,y,.77,.48,math.pi/2)
        box('workstation partition',(2.03,y,.985),(.028,.76,.43),white,.006)
    # RGB PC tower faces aisle close to the doorway, as in photos 5/6.
    box('RGB desktop tower',(2.04,1.76,.23),(.24,.45,.56),plastic,.012)
    rgbglass=material('PC tinted panel',(.012,.016,.025),.14,.25)
    box('tower acrylic side',(1.914,1.76,.25),(.008,.39,.47),rgbglass,.002)
    for j,col in enumerate(((.03,.03,1.),(.08,.55,1.),(.36,.015,1.))):
        ledmat=material('PC glow '+str(j),col,.4,emission=1.8)
        rod('RGB fan',(1.906,1.67,.09+j*.13),(1.912,1.67,.09+j*.13),.043,ledmat,'decoration',24)
        rod('fan dark hub',(1.902,1.67,.09+j*.13),(1.906,1.67,.09+j*.13),.025,plastic,'decoration',24)
    carton('right aisle open box',(1.30,2.13,-.05),(.53,.52,.42),True)
    # Papers, jars, books and a cloth bag; local arrangement inferred from photos.
    for j in range(14):
        x=rng.uniform(1.32,2.5);y=rng.uniform(1.66,4.0)
        box('desk papers',(x,y,.779+j*.002),(.17,.22,.003),paper,rotation=rng.uniform(-.4,.4))
    for j,(x,y) in enumerate(((1.5,2.63),(2.37,2.26),(1.56,3.55))):
        rod('desk jar',(x,y,.79),(x,y,.97),.046,cardboard if j==0 else white,vertices=20)
        rod('jar lid',(x,y,.968),(x,y,.984),.049,blue if j==0 else darksteel,vertices=20)
    box('green document folder',(1.54,1.80,.88),(.24,.17,.025),material('green folder',(.21,.34,.17),.7),.003,rotation=.35)
    bag=box('cream shopping bag',(2.40,1.9,.91),(.37,.36,.25),material('cream bag',(.75,.70,.52),.93),.07)
    for j in range(4):
        pts=[(2.13+j*.025,1.72,.75),(2.17+j*.015,1.56,.37),(2.06+j*.02,1.40,-.045)]
        pts += [(2.03+.12*math.cos(k*.16+j),1.18+.09*math.sin(k*.16),-.046) for k in range(42)]
        cable('desk loose cables',pts,black,.003)

    # Red-bordered wall notice: text-like rule layout rather than fictitious readable content.
    px,py,pz=left+.013,2.35,1.92
    box('notice silver frame',(px,py,pz),(.028,1.46,1.10),steel,.005,'decoration')
    box('notice paper',(px+.017,py,pz),(.008,1.42,1.06),paper,category='decoration')
    for z,hh in ((2.38,.13),(1.47,.13)):
        box('notice red band',(px+.023,py,z),(.004,1.41,hh),red,category='decoration')
    box('notice yellow footer stripe',(px+.026,py,1.46),(.002,1.4,.028),material('notice yellow',(.84,.66,.035),.7),category='decoration')
    for col in range(3):
        for row in range(26):
            length=.32*rng.uniform(.50,1.)
            box('notice printed line',(px+.026,py-.48+col*.47,2.265-row*.027),(.002,length,.004),darksteel,category='decoration')
    # Small desk figurine as visible on the window-side stand.
    rod('figurine torso',(.51,4.30,1.16),(.51,4.30,1.32),.025,red,'decoration')
    for side in (-1,1):
        rod('figurine leg',(.51+side*.02,4.30,1.18),(.51+side*.05,4.30,1.04),.012,darksteel,'decoration')
        rod('figurine arm',(.51+side*.02,4.30,1.29),(.51+side*.08,4.30,1.36),.012,darksteel,'decoration')

    # Read-only conservative clearance audit against each moving body's swept AABB.
    owners={}
    for role,records in data['frames'][0]['bodies'].items():
        for name in records:
            key=name if role=='door' else 'robot/'+name
            if key in bpy.data.objects:owners[key]=(role,name)
    moving=[]
    matrices={}
    for obj in bpy.data.objects:
        if obj.type!='MESH' or obj in created:continue
        parent=obj.parent
        while parent is not None and parent.name not in owners:parent=parent.parent
        if parent is None:continue
        role,name=owners[parent.name]
        relative=parent.matrix_world.inverted()@obj.matrix_world
        points=np.array([(*tuple(relative@Vector(c)),1.) for c in obj.bound_box])
        if parent.name not in matrices:
            matrices[parent.name]=np.array([np.array(convert@pose(f['bodies'][role][name])) for f in data['frames']])
        transformed=np.einsum('tij,kj->tki',matrices[parent.name],points)[...,:3]
        moving.append((obj.name,role,transformed.min(axis=1),transformed.max(axis=1)))
    scene.frame_set(1);bpy.context.view_layer.update()
    warnings=[];minimum=float('inf');checks=[]
    for obj in solid:
        # Architectural doorway walls intentionally abut the retained frame;
        # audit movable robot only there, door+robot for furniture.
        coords=np.array([tuple(obj.matrix_world@Vector(c)) for c in obj.bound_box])
        lo,hi=coords.min(axis=0),coords.max(axis=0)
        best=float('inf');intersections=[]
        for name,role,a,b in moving:
            if role=='door' and obj.get('photo_category') in ('wall','ceiling'):continue
            distances=np.linalg.norm(np.maximum(np.maximum(lo-b,a-hi),0),axis=1)
            d=float(distances.min());best=min(best,d)
            if d==0:
                intersections.append(dict(body=name,role=role,frames=int((distances==0).sum())))
        minimum=min(minimum,best)
        if intersections:warnings.append(dict(object=obj.name,intersections=intersections))
        checks.append(dict(name=obj.name,minimum_aabb_gap_m=best))
    references=[]
    photo_dir=Path(room['reference_dir'])
    for i in range(1,7):
        p=photo_dir/f'{i}.jpg'
        references.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    asset_after=asset_signature()
    assert asset_before==asset_after,'Door/robot geometry or materials unexpectedly changed'
    assert not warnings,('New room intersects recorded moving-body bounds',warnings)
    report=dict(scene_variant='photo_office_20261008',settings=settings,
        original_assets_unchanged=True,original_assets_sha256=asset_before,
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),reference_images=references,
        duplicate_reference_images=['3.jpg','4.jpg'],inferred_dimensions=True,
        dimensions_m=dict(office_width=right-left,office_depth=back,office_ceiling=h_office,entrance_ceiling=h_entry),
        lights=light_report,light_multiplier=light_multiplier,emissive_materials=emission_report,
        objects=len(created),frames_checked=len(data['frames']),
        clearance_method='Per-recorded-frame moving visual-body AABBs vs added static mesh AABBs. Architecture abutting original frame excludes door checks. Floor/decals/wires excluded; not contact simulation.',
        minimum_aabb_gap_m=minimum,clearance_warnings=warnings,object_clearance=checks,render_only=True,
        limitations=['Dimensions, unseen entrance wall, concealed furniture and light wattages estimated from images.',
                     'Props are reconstructed approximations, not photogrammetric scans.',
                     'No new physics collision actors; original door/robot motion retained.'])
    (Path(out)/'room_clearance.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('PHOTO_ROOM_BUILT',len(created),'CLEARANCE_WARNINGS',len(warnings),flush=True)
    # Third view is a human-height walkthrough to show the reconstructed room.
    overview=bpy.data.objects['Replay overview']
    overview.animation_data_clear();overview.rotation_mode='QUATERNION'
    overview.data.lens=19;overview.data.sensor_width=36;overview.data.clip_start=.03
    for frame,loc,target in [
        (1,(.28,-2.75,1.60),(0,0,1.42)),(240,(.18,-1.8,1.60),(0,0,1.25)),
        (350,(.15,-1.8,1.60),(0,.3,1.25)),(420,(.13,-.45,1.60),(.5,3.8,1.25)),
        (480,(.10,.05,1.60),(.4,4.5,1.00)),(520,(.1,.25,1.60),(.35,4.65,.95))]:
        overview.location=loc;overview.rotation_quaternion=(Vector(target)-overview.location).to_track_quat('-Z','Y')
        overview.keyframe_insert('location',frame=frame);overview.keyframe_insert('rotation_quaternion',frame=frame)
    if overview.animation_data and overview.animation_data.action:
        for fc in overview.animation_data.action.fcurves:
            for kp in fc.keyframe_points:kp.interpolation='LINEAR'
    scene['photo_room_report']=str(Path(out)/'room_clearance.json')
    return report
