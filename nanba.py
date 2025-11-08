###############################################
########### Numba Helper Functions ############
######## (Nanba in tamil means friend) ########
###############################################

import numpy as np
import math
from numba import jit, prange

STEPS = 60000
D_LAMBDA = 1e-3
ESCAPE_R = 1000
RAY_SPEED = 100 

@jit(nopython=True, fastmath=True, cache=True)
def geodesic_rhs(r, theta, dr, dtheta, dphi, rt):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    d1_r = dr
    d1_theta = dtheta
    d1_phi = dphi

    d2_t = 0 # No Acceleration in Time Dimension
    
    d2_r = r * (dtheta*dtheta + sin_theta*sin_theta*dphi*dphi)
    
    common_term_r = r / (r*r + rt*rt) if (r*r + rt*rt) > 1e-9 else 0.0
    
    d2_theta = -2.0 * common_term_r * dr * dtheta
    d2_theta += sin_theta * cos_theta * dphi*dphi
    
    d2_phi = -2.0 * common_term_r * dr * dphi - (2.0 * cos_theta / sin_theta) * dtheta * dphi
    # if sin_theta > 1e-6:
    #     d2_phi -= (2.0 * cos_theta / sin_theta) * dtheta * dphi
        
    return d1_r, d1_theta, d1_phi, d2_r, d2_theta, d2_phi

@jit(nopython=True, fastmath=True, cache=True)
def rk4_step(r, theta, phi, dr, dtheta, dphi, rt, dL):
    dL_half = 0.5 * dL
    dL_sixth = dL / 6.0
    
    (k1_r, k1_theta, k1_phi, 
     k1_dr, k1_dtheta, k1_dphi) = geodesic_rhs(r, theta, dr, dtheta, dphi, rt)

    r_k2 = r + k1_r * dL_half
    theta_k2 = theta + k1_theta * dL_half
    phi_k2 = phi + k1_phi * dL_half
    dr_k2 = dr + k1_dr * dL_half
    dtheta_k2 = dtheta + k1_dtheta * dL_half
    dphi_k2 = dphi + k1_dphi * dL_half
    
    (k2_r, k2_theta, k2_phi, 
     k2_dr, k2_dtheta, k2_dphi) = geodesic_rhs(r_k2, theta_k2, dr_k2, dtheta_k2, dphi_k2, rt)

    r_k3 = r + k2_r * dL_half
    theta_k3 = theta + k2_theta * dL_half
    phi_k3 = phi + k2_phi * dL_half
    dr_k3 = dr + k2_dr * dL_half
    dtheta_k3 = dtheta + k2_dtheta * dL_half
    dphi_k3 = dphi + k2_dphi * dL_half
    
    (k3_r, k3_theta, k3_phi, 
     k3_dr, k3_dtheta, k3_dphi) = geodesic_rhs(r_k3, theta_k3, dr_k3, dtheta_k3, dphi_k3, rt)

    r_k4 = r + k3_r * dL
    theta_k4 = theta + k3_theta * dL
    phi_k4 = phi + k3_phi * dL
    dr_k4 = dr + k3_dr * dL
    dtheta_k4 = dtheta + k3_dtheta * dL
    dphi_k4 = dphi + k3_dphi * dL
    
    (k4_r, k4_theta, k4_phi, 
     k4_dr, k4_dtheta, k4_dphi) = geodesic_rhs(r_k4, theta_k4, dr_k4, dtheta_k4, dphi_k4, rt)

    # Final Combination:
    # new_y = y + (k1 + 2*k2 + 2*k3 + k4) * (h/6)
    
    r_new = r + (k1_r + 2.0*k2_r + 2.0*k3_r + k4_r) * dL_sixth
    theta_new = theta + (k1_theta + 2.0*k2_theta + 2.0*k3_theta + k4_theta) * dL_sixth
    phi_new = phi + (k1_phi + 2.0*k2_phi + 2.0*k3_phi + k4_phi) * dL_sixth
    
    dr_new = dr + (k1_dr + 2.0*k2_dr + 2.0*k3_dr + k4_dr) * dL_sixth
    dtheta_new = dtheta + (k1_dtheta + 2.0*k2_dtheta + 2.0*k3_dtheta + k4_dtheta) * dL_sixth
    dphi_new = dphi + (k1_dphi + 2.0*k2_dphi + 2.0*k3_dphi + k4_dphi) * dL_sixth

    sin_theta_new = math.sin(theta_new)
    x = r_new * sin_theta_new * math.cos(phi_new)
    y = r_new * sin_theta_new * math.sin(phi_new)
    z = r_new * math.cos(theta_new)
    
    return x, y, z, r_new, theta_new, phi_new, dr_new, dtheta_new, dphi_new

@jit(nopython=True, fastmath=True, cache=True)
def euler_step(r, theta, phi, dr, dtheta, dphi, rt, dL):
    # TODO: Also try rk4 - DID THAT
    d1_r, d1_theta, d1_phi, d2_r, d2_theta, d2_phi = geodesic_rhs(
        r, theta, dr, dtheta, dphi, rt
    )
    r_new = r + d1_r * dL
    theta_new = theta + d1_theta * dL
    phi_new = phi + d1_phi * dL
    dr_new = dr + d2_r * dL
    dtheta_new = dtheta + d2_theta * dL
    dphi_new = dphi + d2_phi * dL
    sin_theta_new = math.sin(theta_new)
    x = r_new * sin_theta_new * math.cos(phi_new)
    y = r_new * sin_theta_new * math.sin(phi_new)
    z = r_new * math.cos(theta_new)
    return x, y, z, r_new, theta_new, phi_new, dr_new, dtheta_new, dphi_new

@jit(nopython=True, fastmath=True, cache=True)
def init_ray(pos, dir, cam_universe):
    x, y, z = pos[0], pos[1], pos[2]
    dx, dy, dz = dir[0], dir[1], dir[2]

    r_abs = math.sqrt(x*x + y*y + z*z)

    if cam_universe == 1:
        r = r_abs
    else:
        r = -r_abs # Start the ray in Universe 2

    r_safe = r_abs
    # if r_safe < 1e-9:
    #     r_safe = 1e-9

    z_over_r = z / r_safe
    if z_over_r > 1.0: z_over_r = 1.0
    if z_over_r < -1.0: z_over_r = -1.0

    theta = math.acos(z_over_r)
    phi = math.atan2(y, x)

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    dr = sin_theta*cos_phi*dx + sin_theta*sin_phi*dy + cos_theta*dz
    dtheta = (cos_theta*cos_phi*dx + cos_theta*sin_phi*dy - sin_theta*dz) / r_safe
    dphi = (-sin_phi*dx + cos_phi*dy) / (r_safe * sin_theta)

    # if r_safe > 1e-9:
    #     dtheta = (cos_theta*cos_phi*dx + cos_theta*sin_phi*dy - sin_theta*dz) / r_safe
    # else:
    #     dtheta = 0.0

    # if r_safe > 1e-9 and sin_theta > 1e-6:
    #     dphi = (-sin_phi*dx + cos_phi*dy) / (r_safe * sin_theta)
    # else:
    #     dphi = 0.0
    
    # if cam_universe == 2:
    #     dr = -dr
    #     dtheta = -dtheta

    dr *= RAY_SPEED
    dtheta *= RAY_SPEED
    dphi *= RAY_SPEED
    
    return x, y, z, r, theta, phi, dr, dtheta, dphi

@jit(nopython=True, fastmath=True, cache=True)
def sample_background_planar(lensed_dir_x, lensed_dir_y, lensed_dir_z,
                           cam_right, cam_up, cam_forward,
                           bg_data, bg_width, bg_height,
                           tan_half_fov, aspect):
    
    lensed_dir = np.array([lensed_dir_x, lensed_dir_y, lensed_dir_z])
    
    cam_space_x = np.dot(lensed_dir, cam_right)
    cam_space_y = np.dot(lensed_dir, cam_up)
    cam_space_z = np.dot(lensed_dir, cam_forward)
    
    if cam_space_z < 1e-6: 
        return (0, 0, 0) 

    u_proj = cam_space_x / cam_space_z
    v_proj = cam_space_y / cam_space_z
    
    scale = 0.5 
    u = (u_proj * scale) + 0.5
    v = (v_proj * scale) + 0.5
    
    px = int(u * bg_width) % bg_width
    py = int(v * bg_height) % bg_height
    
    return (bg_data[py, px, 0], bg_data[py, px, 1], bg_data[py, px, 2])

@jit(nopython=True, fastmath=True, cache=True)
def sample_background_spherical(lensed_dir_x, lensed_dir_y, lensed_dir_z,
                                bg_data, bg_width, bg_height):
    """
    Samples a 360-degree equirectangular (spherical) image.
    """
    
    phi = math.atan2(lensed_dir_y, lensed_dir_x) # range -pi to +pi
    
    theta = math.acos(lensed_dir_z) # range 0 to +pi
    
    u = (phi + math.pi) / (2.0 * math.pi) # map [0, 1]
    v = theta / math.pi                  # map [0, 1]
    
    px = int(u * bg_width)
    py = int(v * bg_height)
    
    if px >= bg_width: px = bg_width - 1
    if py >= bg_height: py = bg_height - 1
        
    return (bg_data[py, px, 0], bg_data[py, px, 1], bg_data[py, px, 2])

@jit(nopython=True, fastmath=True, cache=True)
def raywarp_pixel(px, py, compute_width, compute_height, 
                   cam_pos, cam_right, cam_up, cam_forward,
                   tan_half_fov, aspect, rt, cam_universe,
                   space1_image_data, space1_width, space1_height,
                   space2_image_data, space2_width, space2_height):
    
    u_ndc = (2.0 * (px + 0.5) / compute_width - 1.0) * aspect * tan_half_fov
    v_ndc = (1.0 - 2.0 * (py + 0.5) / compute_height) * tan_half_fov
    
    dir_x = u_ndc * cam_right[0] - v_ndc * cam_up[0] + cam_forward[0]
    dir_y = u_ndc * cam_right[1] - v_ndc * cam_up[1] + cam_forward[1]
    dir_z = u_ndc * cam_right[2] - v_ndc * cam_up[2] + cam_forward[2]
    
    dir_norm = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    dir_vec_x = dir_x / dir_norm
    dir_vec_y = dir_y / dir_norm
    dir_vec_z = dir_z / dir_norm

    dir_vec = np.array([dir_vec_x, dir_vec_y, dir_vec_z])

    (x, y, z, r, theta, phi, 
     dr, dtheta, dphi) = init_ray(cam_pos, dir_vec, cam_universe)
    
    escaped_side = 0 # 0 = trapped, 1 = our side, 2 = other side
    
    for i in range(STEPS):
        # (x, y, z, r, theta, phi, dr, dtheta, dphi) = euler_step(r, theta, phi, dr, dtheta, dphi, rt, D_LAMBDA)
        (x, y, z, r, theta, phi, dr, dtheta, dphi) = rk4_step(r, theta, phi, dr, dtheta, dphi, rt, D_LAMBDA)
        
        # print(r)

        if r > ESCAPE_R:
            escaped_side = 1
            break
            
        if r < -ESCAPE_R:
            escaped_side = 2
            break
            
    if escaped_side == 1:
        # Escaped on our side (Universe 1)
        final_dir_norm = math.sqrt(x*x + y*y + z*z)
        lensed_dir_x = x / final_dir_norm
        lensed_dir_y = y / final_dir_norm
        lensed_dir_z = z / final_dir_norm
        
        # return sample_background_planar(
        #     lensed_dir_x, lensed_dir_y, lensed_dir_z, 
        #     cam_right, cam_up, cam_forward,
        #     space1_image_data, space1_width, space1_height,
        #     tan_half_fov, aspect
        # )
        return sample_background_spherical(
            lensed_dir_x, lensed_dir_y, lensed_dir_z, 
            space1_image_data, space1_width, space1_height
        )
        
    elif escaped_side == 2:
        # Escaped on the other side (Universe 2)
        final_dir_norm = math.sqrt(x*x + y*y + z*z)
        lensed_dir_x = x / final_dir_norm
        lensed_dir_y = y / final_dir_norm
        lensed_dir_z = z / final_dir_norm
        
        # return sample_background_planar(
        #     lensed_dir_x, lensed_dir_y, lensed_dir_z, 
        #     cam_right, cam_up, cam_forward,
        #     space2_image_data, space2_width, space2_height,
        #     tan_half_fov, aspect
        # )
        return sample_background_spherical(
           lensed_dir_x, lensed_dir_y, lensed_dir_z, 
            space2_image_data, space2_width, space2_height
        )
    else: 
        # Ray got "trapped" or ran out of steps
        print("If you see this message often, consider increasing STEPS or ESCAPE_R.")
        return (0, 0, 0) # Return black
    
# My definitions:
# Raytrace - when the ray is traced analytically in straight lines until it hits something
# Raymarch - when the ray is traced in small steps, sampling along the way in straight lines
# Raywarp - when the ray is traced in small steps, but a geodesic for two points is no longer a straight line, so the rays warp around spacetime

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def raywarp_kernel(pixels, compute_width, compute_height,
                    cam_pos, cam_right, cam_up, cam_forward,
                    tan_half_fov, aspect, 
                    rt, cam_universe,
                    space1_image_data, space1_width, space1_height,
                    space2_image_data, space2_width, space2_height):
    for py in prange(compute_height):
        for px in range(compute_width):
            r, g, b = raywarp_pixel(px, py, compute_width, compute_height,
                                     cam_pos, cam_right, cam_up, cam_forward,
                                     tan_half_fov, aspect, 
                                     rt, cam_universe,
                                     space1_image_data, space1_width, space1_height,
                                     space2_image_data, space2_width, space2_height)
            pixels[py, px, 0] = r
            pixels[py, px, 1] = g
            pixels[py, px, 2] = b