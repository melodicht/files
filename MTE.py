import os
import argparse
import numpy as np
import btypes as bt
import texdump
import time
import subprocess

#------------------------------------------------------------------------------

class Header(bt.Struct(
  bt.Field(bt.uint32,'magic'), # magic == 0x04B40000
  bt.Field(bt.uint16,'num_faces'),
  bt.Padding(2,b'\x00'),
  bt.Field(bt.uint16,'num_nodes'), # num_nodes == num_joints
  bt.Field(bt.uint16,'num_shape_packets'),
  bt.Field(bt.uint16,'num_weighted_matrices'),
  bt.Field(bt.uint16,'num_joints'),
  bt.Field(bt.uint16,'num_vertices'),
  bt.Field(bt.uint16,'num_normals'),
  bt.Field(bt.uint16,'num_colors'),
  bt.Field(bt.uint16,'num_texcoords'),
  bt.Padding(8,b'\x00'),
  bt.Field(bt.uint16,'num_textures'),
  bt.Padding(2,b'\x00'),
  bt.Field(bt.uint16,'num_texobjs'),
  bt.Field(bt.uint16,'num_draw_elements'), # num_draw_elements == num_shapes
  bt.Field(bt.uint16,'num_materials'),
  bt.Field(bt.uint16,'num_shapes'),
  bt.Padding(4,b'\x00'),
  bt.Field(bt.uint32,'node_offset'),
  bt.Field(bt.uint32,'shape_packet_offset'),
  bt.Field(bt.uint32,'matrix_offset'),
  bt.Field(bt.uint32,'weight_offset'),
  bt.Field(bt.uint32,'joint_index_offset'),
  bt.Field(bt.uint32,'num_weights_offset'),
  bt.Field(bt.uint32,'vertex_offset'),
  bt.Field(bt.uint32,'normal_offset'),
  bt.Field(bt.uint32,'color_offset'),
  bt.Field(bt.uint32,'texcoord_offset'),
  bt.Padding(8,b'\x00'),
  bt.Field(bt.uint32,'texture_location_offset'),
  bt.Padding(4,b'\x00'),
  bt.Field(bt.uint32,'material_offset'),
  bt.Field(bt.uint32,'texobj_offset'),
  bt.Field(bt.uint32,'shape_offset'),
  bt.Field(bt.uint32,'draw_element_offset'),
  bt.Padding(8,b'\x00'))): pass


class Vector(bt.Struct(
  bt.Field(bt.float32,'x'),
  bt.Field(bt.float32,'y'),
  bt.Field(bt.float32,'z'))): pass


class TextureCoordinate(bt.Struct(
  bt.Field(bt.float32,'s'),
  bt.Field(bt.float32,'t'))): pass


class Color(bt.Struct(
  bt.Field(bt.uint8,'r'),
  bt.Field(bt.uint8,'g'),
  bt.Field(bt.uint8,'b'),
  bt.Field(bt.uint8,'a'))): pass


class Matrix:

  IDENTITY = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

  @staticmethod
  def unpack(stream):
    row0 = [stream >> bt.float32 for i in range(4)]
    row1 = [stream >> bt.float32 for i in range(4)]
    row2 = [stream >> bt.float32 for i in range(4)]
    return np.matrix([row0,row1,row2,[0,0,0,1]])


class Node(bt.Struct(
  bt.Field(bt.uint16,'unknown0'),
  bt.Field(bt.uint16,'unknown1'), # unknown1 in (0,1)
  bt.Field(bt.uint16,'unknown2'),
  bt.Field(bt.uint16,'unknown3'), # unknown3 in (0,1)
  bt.Field(bt.uint16,'unknown4'),
  bt.Field(bt.uint16,'unknown5'),
  bt.Padding(4,b'\x00'))): pass


class TextureHeader(bt.Struct(
  bt.Field(bt.uint8,'format'),
  bt.Padding(1,b'\x00'),
  bt.Field(bt.uint16,'width'),
  bt.Field(bt.uint16,'height'),
  bt.Padding(26,b'\x00'))): pass


class TexObj(bt.Struct(
  bt.Field(bt.uint16,'texture_index'),
  bt.Padding(2),
  bt.Field(bt.uint8,'unknown1'), # unknown1 in (0,1,2)
  bt.Field(bt.uint8,'unknown2'), # unknown2 in (0,1,2)
  bt.Field(bt.uint8,'unknown3'), # unknown3 in (0,1,4)
  bt.Padding(1,b'\x00'))): pass


class TevStage(bt.Struct(
  bt.Field(bt.uint16,'unknown0'), # unknown0 in (0,0x200)
  bt.Field(bt.uint16,'texobj_index'),
  bt.Field(bt.Array(bt.float32,7),'unknown1'))): pass


class Material(bt.Struct(
  bt.Field(Color,'color'),
  bt.Field(bt.uint16,'unknown1'), # unknown1 in (0,1), number of colors?
  bt.Field(bt.uint8,'unknown2'), # unknown2 in (0,1,2,4)
  bt.Field(bt.uint8,'num_tev_stages'), # unknown3 in (0,1,2)
  bt.Field(bt.uint8,'unknown4'), # unknown4 in (0,1)
  bt.Padding(23,b'\x00'),
  bt.Field(bt.Array(TevStage,8),'tev_stages'))): pass


class Shape(bt.Struct(
  bt.Field(bt.uint8,'unknown0'), # unknown0 in (0,1,3)
  bt.Field(bt.uint8,'unknown1'), # unknown1 == 0
  bt.Field(bt.uint8,'unknown2'),
  bt.Field(bt.uint8,'unknown3'), # unknown3 in (0,1)
  bt.Field(bt.uint16,'num_packets'),
  bt.Field(bt.uint16,'first_packet'))): pass


class ShapePacket(bt.Struct(
  bt.Field(bt.uint32,'data_offset'),
  bt.Field(bt.uint32,'data_size'),
  bt.Field(bt.uint16,'unknown0'), # unknown0 in (0,2)
  bt.Field(bt.uint16,'num_matrix_indices'),
  bt.Field(bt.Array(bt.uint16,10),'matrix_indices'))): pass


class DrawElement(bt.Struct(
  bt.Field(bt.uint16,'material_index'),
  bt.Field(bt.uint16,'shape_index'))): pass

#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Extract/Inject images into MDL files')
parser.add_argument('inout',metavar='i/o')
parser.add_argument('ifile',metavar='input.mdl')
parser.add_argument('img',nargs='?',metavar='image')
parser.add_argument('texnum',nargs='?',metavar='"texture number"', type=int)
arguments = parser.parse_args()

rootname = os.path.splitext(arguments.ifile)[0]

stream = bt.FileStream(arguments.ifile,'rb',bt.BE)
header = stream >> Header

stream.seek(header.material_offset)
materials = [stream >> Material for i in range(header.num_materials)]

stream.seek(header.texobj_offset)
texobjs = [stream >> TexObj for i in range(header.num_texobjs)]

stream.seek(header.shape_offset)
shapes = [stream >> Shape for i in range(header.num_shapes)]

stream.seek(header.draw_element_offset)
draw_elements = [stream >> DrawElement for i in range(header.num_draw_elements)]

stream.seek(header.shape_packet_offset)
shape_packets = [stream >> ShapePacket for i in range(header.num_shape_packets)]

stream.seek(header.texture_location_offset)
texture_offsets = [stream >> bt.uint32 for i in range(header.num_textures)]

tex_amount = len(texture_offsets)
print('locations:')
if tex_amount == 1:
    tex1 = texture_offsets
if tex_amount == 2:
    tex1, tex2 = texture_offsets
if tex_amount == 3:
    tex1, tex2, tex3 = texture_offsets
if tex_amount == 4:
    tex1, tex2, tex3, tex4 = texture_offsets
if tex_amount == 5:
    tex1, tex2, tex3, tex4, tex5 = texture_offsets
if tex_amount == 6:
    tex1, tex2, tex3, tex4, tex5, tex6 = texture_offsets
if tex_amount == 7:
    tex1, tex2, tex3, tex4, tex5, tex6, tex7 = texture_offsets
if tex_amount == 8:
    tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8 = texture_offsets
if tex_amount == 9:
    tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9 = texture_offsets
if tex_amount == 10:
    tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9, tex10 = texture_offsets
    
print (texture_offsets)

if tex_amount >= 1:
    print ('texture file 0:')
    print (hex(tex1))
    print (' ')
if tex_amount >= 2:
    print ('texture file 1:')
    print (hex(tex2))
    print (' ')
if tex_amount >= 3:
    print ('texture file 2:')
    print (hex(tex3))
    print (' ')
if tex_amount >= 4:
    print ('texture file 3:')
    print (hex(tex4))
    print (' ')
if tex_amount >= 5:
    print ('texture file 4:')
    print (hex(tex5))
    print (' ')
if tex_amount >= 6:
    print ('texture file 5:')
    print (hex(tex6))
    print (' ')
if tex_amount >= 7:
    print ('texture file 6:')
    print (hex(tex7))
    print (' ')
if tex_amount >= 8:
    print ('texture file 7:')
    print (hex(tex8))
    print (' ')
if tex_amount >= 9:
    print ('texture file 8:')
    print (hex(tex9))
    print (' ')
if tex_amount >= 10:
    print ('texture file 9:')
    print (hex(tex110))

stream.seek(header.vertex_offset)
vertices = [stream >> Vector for i in range(header.num_vertices)]

stream.seek(header.normal_offset)
normals = [stream >> Vector for i in range(header.num_normals)]

stream.seek(header.texcoord_offset)
texcoords = [stream >> TextureCoordinate for i in range(header.num_texcoords)]

stream.seek(header.matrix_offset)
global_matrix_table = [(stream >> Matrix).I for i in range(header.num_joints)] + [Matrix.IDENTITY]*header.num_weighted_matrices

if arguments.inout == 'i':
    # Import textures
    for i in range(arguments.texnum+1):
        texture_offset = texture_offsets[i]
        stream.seek(texture_offset)
        texture_header = stream >> TextureHeader
        
        
        if texture_header.format == 0x03:
            injtype = 1
            truetype = 'I4'
        elif texture_header.format == 0x04:
            injtype = 2
            truetype = 'I8'
        elif texture_header.format == 0x06:
            injtype = 3
            truetype = 'IA8'
        elif texture_header.format == 0x07:
            injtype = 4
            truetype = '565'
        elif texture_header.format == 0x08:
            injtype = 5
            truetype = '5A3'
        elif texture_header.format == 0x0A:
            injtype = 6
            truetype = 'cmpr'
        else:
            raise Exception('unsuported texture format: 0x{:02X}'.format(texture_header.format))
    batdata = 'TGAtoBTI.exe '+'-'+truetype+' '+arguments.img+' converted.bti'
    os.path.dirname(os.path.realpath(__file__))
    subprocess.call(batdata, shell=True)
    time.sleep(2)
    ASCIIdata=open('asciidata','rb')
    ASCIIthing=bytes(ASCIIdata.read()[injtype])
    MDLfile=open(arguments.ifile,'rb')
    BTIfile=open('converted.bti','rb')
    BTIdata=bytes(BTIfile.read())
    MDLdata=MDLfile.read()
    MDLfile.close()
    MDLfile=open('new.mdl','wb')
    offset=texture_offsets[arguments.texnum]
    fileendoffset=texture_offsets[arguments.texnum+1]+len(BTIfile.read())+1
    MDLfile.write(bytes(MDLdata[:offset]))
    MDLfile.write(bytes(ASCIIthing))
    MDLfile.write(BTIdata[len(BTIdata)-1:])
    endoffset=len(MDLdata[:offset-1])+len(BTIfile.read())
    MDLfile.write(bytes(MDLdata[len(MDLdata)-endoffset:]))

if arguments.inout == 'o':
    # Export textures
    print(' ')
    print('image types:')
    print(' ')
    for texture_index,texture_offset in enumerate(texture_offsets):
        stream.seek(texture_offset)
        texture_header = stream >> TextureHeader
  

        if texture_header.format == 0x03:
            image = texdump.unpack_i4(stream,texture_header.width,texture_header.height)
            print('file type:')
            print('i4 / 03')
        elif texture_header.format == 0x04:
            image = texdump.unpack_i8(stream,texture_header.width,texture_header.height)
            print('file type:')
            print('i8 / 04')
        elif texture_header.format == 0x06:
            image = texdump.unpack_ia8(stream,texture_header.width,texture_header.height)
            print('file type:')
            print('ia8 / 06')
        elif texture_header.format == 0x07:
            image = texdump.unpack_rgb565(stream,texture_header.width,texture_header.height)
            print('file type:')
            print('rgb565 / 07')
        elif texture_header.format == 0x08:
            image = texdump.unpack_rgb5a3(stream,texture_header.width,texture_header.height)
            print('file type:')
            print('rgb5a3 / 08')
        elif texture_header.format == 0x0A:
            image = texdump.unpack_cmpr(stream,texture_header.width,texture_header.height)
            print('file type:')
            print('cmpr / 0A')
        else:
            raise Exception('unsuported texture format: 0x{:02X}'.format(texture_header.format))

        image.save('{}_{}.png'.format(rootname,texture_index))
        print('resolution:')
        print(texture_header.height,texture_header.width)
        print(' ')
