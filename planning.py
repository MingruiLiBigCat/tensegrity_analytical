import math
class COM_scheduler(object):
    def __init__(self, a, b):
        """
        Initialize the COM_scheduler with coefficients a and b.
        The tensegrity robot is expected to move along ax+by=0
        """
        self.a=a
        self.b=b
        self.slope, self.intercept = self.__line_equation(a, b)

    def __line_equation(self,a, b):
        if b == 0:  
            return (float('inf'), 0)
        slope = -a / b
        intercept = 0
        return (slope, intercept)
    
    def __distance_to_origin(self,x, y):
        """Calculate Euclidean distance from point to origin"""
        return -y

    def __distance_to_line(self,a, b, x, y):
        """Calculate signed distance from point (x,y) to line ax + by = 0"""
        denominator = math.sqrt(a**2 + b**2)
        if denominator == 0:
            return 0  # invalid line, distance is 0
        return (a*x + b*y) / denominator

    def __triangle_edges(self,p1, p2, p3):
        """Return coefficients (a,b,c) for equations of triangle edges (ax + by + c = 0)"""
        edges = []
        # Edge 1: p2-p3
        a = p3[1] - p2[1]
        b = p2[0] - p3[0]
        c = p3[0]*p2[1] - p2[0]*p3[1]
        edges.append((a, b, c))
        
        # Edge 2: p1-p3
        a = p1[1] - p3[1]
        b = p3[0] - p1[0]
        c = p1[0]*p3[1] - p3[0]*p1[1]
        edges.append((a, b, c))
        
        return edges
    
    def get_COM(self,x0,y0,x1,y1,x2,y2,x3,y3,allowance=0.03):
        """
        Calculate the foot of the perpendicular from point (x0, y0) to the line
        defined by the triangle vertices (x1, y1), (x2, y2), (x3, y3).
        The function returns the coordinates of the foot and the maximum difference
        between the distance to the origin and the signed distance to the direction line.
        
        Parameters:
        a_dir (float): Coefficient for the direction line.
        b_dir (float): Coefficient for the direction line.
        x0 (float): x-coordinate of current COM.
        y0 (float): y-coordinate of current COM.
        x2 (float): x-coordinate of the one-side vertex.
        y2 (float): y-coordinate of the one-side vertex.
        """
        # Get equations for triangle edges
        edges = self.__triangle_edges((x1, y1), (x2, y2), (x3, y3))
        
        max_diff = -float('inf')
        result = None
        for i, (a, b, c) in enumerate(edges, 1):
            # Calculate foot of perpendicular
            denominator = a**2 + b**2
            if denominator == 0:
                continue  # skip zero-length edges
            # Foot coordinates
            x_foot = (b*(b*x0 - a*y0) - a*c) / denominator
            y_foot = (a*(-b*x0 + a*y0) - b*c) / denominator
            x_foot += allowance * a / math.sqrt(denominator)
            y_foot += allowance * b / math.sqrt(denominator)
            
            # Distance to origin
            dist_origin = self.__distance_to_origin(x_foot, y_foot)
            
            # Signed distance to direction line
            dist_direction = self.__distance_to_line(self.a, self.b, x_foot, y_foot)
            #print("Planner parameter:",x_foot, y_foot, dist_origin, dist_direction)
            # Calculate the difference
            diff = dist_origin #- abs(dist_direction)
            
            # Update maximum difference
            if diff > max_diff:
                max_diff = diff
                result = {
                    'edge': i,
                    'foot': (x_foot, y_foot),
                    'dist_origin': dist_origin,
                    'dist_direction': dist_direction,
                    'difference': diff
                }
        
        # Output results
        return result["foot"],result