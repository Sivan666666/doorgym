import copy
from pathlib import Path
import sys
import shutil
import subprocess
import re
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'float_ik'))
import rgb_robot_materials as rm
ASSET = ROOT/'data/asset/a2wz1'


class RobotMaterials(unittest.TestCase):
    def test_legacy_noop(self):
        rm.prepare_robot_visuals('/does_not_exist', ['missing'], None)
        rm.prepare_robot_visuals('/does_not_exist', ['missing'], SimpleNamespace(config={}))

    def test_stl_black_mount_preserves_vertices(self):
        source = ASSET/'meshes/wrist_mount_up.STL'
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'mount.obj'
            result = rm.decorate_stl(source,dest,'wrist_down_mount')
            self.assertEqual(set(result['material_faces']),{'black'})
            data = source.read_bytes()
            facets=np.frombuffer(data,dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),offset=84)
            actual=np.asarray([list(map(float,row.split()[1:])) for row in dest.read_text().splitlines() if row.startswith('v ')],np.float32)
            np.testing.assert_array_equal(actual,facets['v'].reshape(-1,3))
            wheel = rm.decorate_stl(ASSET/'meshes/left_front_Link4.STL',Path(tmp)/'wheel.obj','FL_wheel')
            self.assertEqual(set(wheel['material_faces']),{'rubber'})

    def test_dae_positions_normals_faces_preserved(self):
        def vertices_faces(root):
            mesh=root.find('.//c:geometry/c:mesh',rm.N)
            arrays={x.get('id'):x.find('c:float_array',rm.N).text for x in mesh.findall('c:source',rm.N)
                    if not x.get('id').endswith('-label-uv')}
            faces=[]
            for tri in mesh.findall('c:triangles',rm.N):
                inputs=tri.findall('c:input',rm.N)
                stride=max(int(i.get('offset')) for i in inputs)+1
                data=np.fromstring(tri.find('c:p',rm.N).text,sep=' ',dtype=int).reshape(-1,3,stride)
                original=[int(i.get('offset')) for i in inputs if i.get('semantic')!='TEXCOORD']
                faces.extend(tuple(row.flat) for row in data[:,:,original])
            return arrays, sorted(faces)
        for name in sorted(p.name for p in (ASSET/'meshes').glob('*.dae')):
            with tempfile.TemporaryDirectory() as tmp:
                source=ASSET/'meshes'/name
                dest=Path(tmp)/name
                rm.decorate_dae(source,dest)
                self.assertEqual(vertices_faces(ET.parse(source)),vertices_faces(ET.parse(dest)))
                self.assertEqual([ET.tostring(x) for x in ET.parse(source).findall('.//c:node/c:matrix',rm.N)],
                                 [ET.tostring(x) for x in ET.parse(dest).findall('.//c:node/c:matrix',rm.N)])

    def test_dae_importer_element_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'z1_Link02.dae'
            rm.decorate_dae(ASSET/'meshes'/dest.name,dest)
            root=ET.parse(dest).getroot()
            self.assertEqual(list(root)[-1].tag,rm.q('scene'))
            for mesh in root.findall('.//c:geometry/c:mesh',rm.N):
                children=list(mesh)
                vertex_index=children.index(mesh.find('c:vertices',rm.N))
                self.assertTrue(all(children.index(s)<vertex_index for s in mesh.findall('c:source',rm.N)))
                for tri in mesh.findall('c:triangles',rm.N):
                    children=list(tri)
                    p_index=children.index(tri.find('c:p',rm.N))
                    self.assertTrue(all(children.index(i)<p_index for i in tri.findall('c:input',rm.N)))

    def test_all_dae_actual_assimp_import(self):
        assimp=shutil.which('assimp')
        if not assimp:
            self.skipTest('Assimp CLI required for actual visual mesh import test')
        def info(path):
            result=subprocess.run([assimp,'info',str(path)],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            def field(name):
                return re.search(r'^'+re.escape(name)+r'\s+(.+)$',result.stdout,re.M).group(1)
            return {key:field(key) for key in ('Faces:','Minimum point','Maximum point')}
        with tempfile.TemporaryDirectory() as tmp:
            for source in sorted((ASSET/'meshes').glob('*.dae')):
                with self.subTest(mesh=source.name):
                    dest=Path(tmp)/source.name
                    rm.decorate_dae(source,dest)
                    self.assertEqual(info(source),info(dest))

    def test_base_upper_shell_matches_silver_legs(self):
        source=ASSET/'meshes/base_link.STL'
        facets=np.frombuffer(source.read_bytes(),dtype=np.dtype([
            ('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]),offset=84)
        xyz=facets['v'].reshape(-1,3)
        threshold=xyz[:,2].min()+.64*np.ptp(xyz[:,2])
        top=facets['v'].mean(1)[:,2] > threshold
        self.assertTrue(top.any())
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'base.obj'
            rm.decorate_stl(source,dest,'base_link')
            labels=[]
            for row in dest.read_text().splitlines():
                if row.startswith('usemtl '):
                    material=row.split()[1]
                elif row.startswith('f '):
                    labels.append(material)
            self.assertTrue(all(m=='silver' for m in np.asarray(labels)[top]))

    def test_nonvisual_urdf_and_mounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=ET.parse(ASSET/'urdf/a2wz1.urdf')
            for mesh in source.findall('.//mesh'):
                mesh.set('filename',str((ASSET/'urdf'/mesh.get('filename')).resolve()))
            path=Path(tmp)/'robot.urdf'
            source.write(path)
            runtime=SimpleNamespace(config={'robot_appearance':{'enabled':True}},metadata={})
            rm.prepare_robot_visuals(tmp,['robot.urdf'],runtime)
            def strip(tree):
                root=copy.deepcopy(tree.getroot())
                for link in root.findall('link'):
                    for visual in link.findall('visual'):
                        link.remove(visual)
                return ET.tostring(root)
            self.assertEqual(strip(source),strip(ET.parse(path)))
            parts=runtime.metadata['robot_appearance']['parts']
            mounts=[p for p in parts if p['link'] in ('wrist_up_mount','wrist_down_mount')]
            self.assertEqual(len(mounts),2)
            self.assertTrue(all(set(p['material_faces'])=={'black'} for p in mounts))
            final = ET.parse(path)
            for link in final.findall('link'):
                if link.get('name','').endswith('_wheel'):
                    self.assertFalse(link.findall('visual/geometry/sphere'))


if __name__=='__main__':
    unittest.main()
