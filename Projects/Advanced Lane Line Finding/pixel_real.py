import numpy as np

class pixel_real():
    def __init__(self, ploty, left_fit_cr, right_fit_cr):
        self.ploty = ploty
        self.left_fit_cr = left_fit_cr
        self.right_fit_cr = right_fit_cr

    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(self.ploty)
        
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*self.left_fit_cr[0]*y_eval*ym_per_pix + self.left_fit_cr[1])**2)**1.5) / np.absolute(2*self.left_fit_cr[0])
        right_curverad = ((1 + (2*self.right_fit_cr[0]*y_eval*ym_per_pix + self.right_fit_cr[1])**2)**1.5) / np.absolute(2*self.right_fit_cr[0])
        
        return float(left_curverad), float(right_curverad)