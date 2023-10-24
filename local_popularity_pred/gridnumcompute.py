# Author: Jiahui
# Project: Iqiyi local popularity prediction
# Time: 2016-09-01 ~ 2017.09
# First edit: 2017.09.27
# Lastest vision: 2017.09.27

'''
    This file is used to compute the grid number
    >> Input file:
        - longitude
        - latitude
    >> Output:
        - grid number
'''
import math

def gridnum(lev, lon, lat):
    lon_min = 116.25 - 0.0065   # select global area of 0.32*0.32 (32km * 32km)
    lon_max = 116.57 - 0.0065
    lat_min = 39.60 - 0.006
    lat_max = 39.92 - 0.006

    '''
        compute grid number; inv = 0.1

    '''

    inv1 = 0.16
    inv = 0.16
    x = lon - 0.0065
    y = lat - 0.006

    c0 = math.floor((x-lon_min)/float(inv))
    r0 = math.floor((y-lat_min)/float(inv))
    n0 = int(math.ceil(r0*((lon_max-lon_min)/float(inv)) + c0))

    inv2 = 0.08
    inv = 0.08
    x = lon - 0.0065 - c0*inv1
    y = lat - 0.006 - r0*inv1

    c1 = math.floor((x-lon_min)/float(inv))
    r1 = math.floor((y-lat_min)/float(inv))
    n1 = int(math.ceil(r1*((lon_max-lon_min)/float(2*inv)) + c1))

    inv3 = 0.04
    inv = 0.04
    x = lon - 0.0065 - c0*inv1 - c1*inv2
    y = lat - 0.006 - r0*inv1 - r1*inv2

    c2 = math.floor((x-lon_min)/float(inv))
    r2 = math.floor((y-lat_min)/float(inv))
    n2 = int(math.ceil(r2*((lon_max-lon_min)/float(pow(2, 2)*inv)) + c2))

    inv4 = 0.02
    inv = 0.02
    x = lon - 0.0065 - c0*inv1 - c1*inv2 - c2*inv3
    y = lat - 0.006 - r0*inv1 - r1*inv2 - r2*inv3

    c3 = math.floor((x-lon_min)/float(inv))
    r3 = math.floor((y-lat_min)/float(inv))
    n3 = int(math.ceil(r3*((lon_max-lon_min)/float(pow(2, 3)*inv)) + c3))

    inv5 = 0.01
    inv = 0.01
    x = lon - 0.0065 - c0*inv1 - c1*inv2 - c2*inv3 - c3*inv4
    y = lat - 0.006 - r0*inv1 - r1*inv2 - r2*inv3 - r3*inv4

    c4 = math.floor((x-lon_min)/float(inv))
    r4 = math.floor((y-lat_min)/float(inv))
    n4 = int(math.ceil(r4*((lon_max-lon_min)/float(pow(2, 4)*inv)) + c4))

    if n0 < 0:
        return -1, -1, -1, -1, -1
    elif lev == 1:
        return n0, -1, -1, -1, -1
    elif lev == 2:
        return n0, n1, -1, -1, -1
    elif lev == 3:
        return n0, n1 ,n2, -1, -1
    elif lev == 4:
        return n0, n1, n2, n3, -1
    elif lev == 5:
        return n0, n1, n2, n3, n4

def non_hier_gridnum(lev, lon, lat):
    lon_min = 116.25 - 0.0065   # select global area of 0.32*0.32 (32km * 32km)
    lon_max = 116.57 - 0.0065
    lat_min = 39.60 - 0.006
    lat_max = 39.92 - 0.006

    '''
        compute grid number; inv = 0.1

    '''

    inv1 = 0.16
    inv = 0.16
    x = lon - 0.0065
    y = lat - 0.006

    c0 = math.floor((x-lon_min)/float(inv))
    r0 = math.floor((y-lat_min)/float(inv))
    n0 = int(math.ceil(r0*((lon_max-lon_min)/float(inv)) + c0))

    inv2 = 0.08
    inv = 0.08
    x = lon - 0.0065 - c0*inv1
    y = lat - 0.006 - r0*inv1

    c1 = math.floor((x-lon_min)/float(inv))
    r1 = math.floor((y-lat_min)/float(inv))
    n1 = int(math.ceil(r1*((lon_max-lon_min)/float(2*inv)) + c1))

    inv3 = 0.04
    inv = 0.04
    x = lon - 0.0065 - c0*inv1 - c1*inv2
    y = lat - 0.006 - r0*inv1 - r1*inv2

    c2 = math.floor((x-lon_min)/float(inv))
    r2 = math.floor((y-lat_min)/float(inv))
    n2 = int(math.ceil(r2*((lon_max-lon_min)/float(pow(2, 2)*inv)) + c2))

    inv4 = 0.02
    inv = 0.02
    x = lon - 0.0065 - c0*inv1 - c1*inv2 - c2*inv3
    y = lat - 0.006 - r0*inv1 - r1*inv2 - r2*inv3

    c3 = math.floor((x-lon_min)/float(inv))
    r3 = math.floor((y-lat_min)/float(inv))
    n3 = int(math.ceil(r3*((lon_max-lon_min)/float(pow(2, 3)*inv)) + c3))

    inv5 = 0.01
    inv = 0.01
    x = lon - 0.0065 - c0*inv1 - c1*inv2 - c2*inv3 - c3*inv4
    y = lat - 0.006 - r0*inv1 - r1*inv2 - r2*inv3 - r3*inv4

    c4 = math.floor((x-lon_min)/float(inv))
    r4 = math.floor((y-lat_min)/float(inv))
    n4 = int(math.ceil(r4*((lon_max-lon_min)/float(pow(2, 4)*inv)) + c4))

    if n0 < 0:
        return -1, -1, -1, -1, -1
    elif lev == 1:
        return n0, -1, -1, -1, -1
    elif lev == 2:
        return n0, n0*4+n1, -1, -1, -1
    elif lev == 3:
        return n0, n0*4+n1, n0*16 + n1*4 + n2, -1, -1
    elif lev == 4:
        return n0, n0*4+n1, n0*16 + n1*4 + n2, n0*64 + n1*16 + n2*4 + n3, -1
    elif lev == 5:
        return n0, n0*4+n1, n0*16 + n1*4 + n2, n0*64 + n1*16 + n2*4 + n3, n0*256 + n1*64 + n2*16 + n3*4 + n4

def ttlgrid(lev, n0, n1, n2, n3, n4):
    '''
        compute total grid number + cumulative grid number
        :param
            - lev: partition depth
        :return
            - total grid number
            - cumulative grid number
    '''
    if lev == 1:
        return 4, n0
    elif lev == 2:
        return 16, n0*4 + n1
    elif lev == 3:
        return 64, n0*16 + n1*4 + n2
    elif lev == 4:
        return 256, n0*64 + n1*16 + n2*4 + n3
    elif lev == 5:
        return 1024, n0*256 + n1*64 + n2*16 + n3*4 + n4

def gnum(lev, lon, lat):
    lon_min = 116.25 - 0.0065   # select global area of 0.32*0.32 (32km * 32km)
    lon_max = 116.57 - 0.0065
    lat_min = 39.60 - 0.006
    lat_max = 39.92 - 0.006

    '''
        compute grid number; inv = 0.1

    '''

    inv1 = 0.16
    inv = 0.16
    x = lon - 0.0065
    y = lat - 0.006

    c0 = math.floor((x-lon_min)/float(inv))
    r0 = math.floor((y-lat_min)/float(inv))
    n0 = int(math.ceil(r0*((lon_max-lon_min)/float(inv)) + c0))

    if lev == 1:
        return n0

    inv2 = 0.08
    inv = 0.08
    x = lon - 0.0065 - c0*inv1
    y = lat - 0.006 - r0*inv1

    c1 = math.floor((x-lon_min)/float(inv))
    r1 = math.floor((y-lat_min)/float(inv))
    n1 = int(math.ceil(r1*((lon_max-lon_min)/float(2*inv)) + c1))

    if lev == 2:
        return n0*4 + n1

    inv3 = 0.04
    inv = 0.04
    x = lon - 0.0065 - c0*inv1 - c1*inv2
    y = lat - 0.006 - r0*inv1 - r1*inv2

    c2 = math.floor((x-lon_min)/float(inv))
    r2 = math.floor((y-lat_min)/float(inv))
    n2 = int(math.ceil(r2*((lon_max-lon_min)/float(pow(2, 2)*inv)) + c2))

    if lev == 3:
        return n0*16 + n1*4 + n2

    inv4 = 0.02
    inv = 0.02
    x = lon - 0.0065 - c0*inv1 - c1*inv2 - c2*inv3
    y = lat - 0.006 - r0*inv1 - r1*inv2 - r2*inv3

    c3 = math.floor((x-lon_min)/float(inv))
    r3 = math.floor((y-lat_min)/float(inv))
    n3 = int(math.ceil(r3*((lon_max-lon_min)/float(pow(2, 3)*inv)) + c3))

    if lev == 4:
        return n0*64 + n1*16 + n2*4 + n3

    inv5 = 0.01
    inv = 0.01
    x = lon - 0.0065 - c0*inv1 - c1*inv2 - c2*inv3 - c3*inv4
    y = lat - 0.006 - r0*inv1 - r1*inv2 - r2*inv3 - r3*inv4

    c4 = math.floor((x-lon_min)/float(inv))
    r4 = math.floor((y-lat_min)/float(inv))
    n4 = int(math.ceil(r4*((lon_max-lon_min)/float(pow(2, 4)*inv)) + c4))

    if lev == 5:
        return n0*256 + n1*64 + n2*16 + n3*4 + n4

def ttl(lev):
    if lev == 1:
        return 4
    elif lev == 2:
        return 16
    elif lev == 3:
        return 64
    elif lev == 4:
        return 256
    elif lev == 5:
        return 1024

def reversegridnum(lev, gnum):
    if lev == 1:
        return [gnum]
    elif lev == 2:
        n0 = gnum / 4
        n1 = gnum % 4
        return n0, n1
    elif lev == 3:
        n0 = gnum / 16
        n1 = (gnum % 16) / 4
        n2 = (gnum % 16) % 4
        return n0, n1, n2
    elif lev == 4:
        n0 = gnum / 64
        n1 = (gnum % 64) / 16
        n2 = ((gnum % 64) % 16) / 4
        n3 = ((gnum % 64) % 16) % 4
        return n0, n1, n2, n3
    elif lev == 5:
        n0 = gnum / 256
        n1 = (gnum % 256) / 64
        n2 = ((gnum % 256) % 64) / 16
        n3 = (((gnum % 256) % 64) % 16) / 4
        n4 = (((gnum % 256) % 64) % 16) % 4
        return n0, n1, n2, n3, n4
