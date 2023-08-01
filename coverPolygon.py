import math
import matplotlib.pyplot as plt

class Vector:
    def __init__(self, x, y, z = 0):
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2) + math.pow(self.z, 2))
    
    def dot(self, vec2):
        return self.x * vec2.x + self.y * vec2.y + self.z * vec2.z
    
    def cross(self, vec2):
        nx = self.y * vec2.z - self.z * vec2.y
        ny = -self.x * vec2.z + self.z * vec2.x
        nz =  self.x * vec2.y - self.y * vec2.x

        return Vector(nx, ny, nz)


class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def magnitude(self):
        return math.sqrt(math.pow(self.x2 - self.x1, 2) + math.pow(self.y2 - self.y1, 2))
    
    def vector(self):
        return (self.x2 - self.x1, self.y2 - self.y1)
    
    def toLine(self):
        a = self.y1 - self.y2
        b = self.x2 - self.x1
        c = self.x1 * self.y2 - self.x2 * self.y1
        return a, b, c
    
    # percent between [0, 1]
    def midpoint(self, percent = 0.5):
        if percent < 0 or percent > 1:
            raise ValueError("percent must be between [0, 1]")
        vt = self.vector()
        return (self.x1 + vt[0] * percent, self.y1 + vt[1] * percent)

    # Find perpendicular distance to a point
    # Source: https://www.geeksforgeeks.org/perpendicular-distance-between-a-point-and-a-line-in-2-d/
    def distToPoint(self, point):
        # first calculate line in ax + by = c form
        a, b, c = self.toLine()
        return abs(a * point[0] + b * point[1] + c) / math.sqrt(a * a + b * b)
    
    def __repr__(self):
        return (f"[x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}]")

"""
Build the counterclockwise convex hull of a polygon.
If a concave polygon is provided, interior vertices will be omitted
@arg points: a list of points defining the polygon
@returns an ordered list defining the convex hull
"""
def convexHull(points):
    # clean up input, remove duplicates
    minPoint = tuple(points[0])
    points = set([tuple(p) for p in points])

    # start wth smallest x & y
    for point in points:
        if point[0] < minPoint[0] or point[0] == minPoint[0] and point[1] < minPoint[1]:
            minPoint = point
    points.remove(minPoint)

    # order points by slope 
    order = [(float('inf') if pt[0] == minPoint[0] else (pt[1] - minPoint[1]) / (pt[0] - minPoint[0]), pt) for pt in points]
    order.sort()

    # clean out points that make hull concave
    ans = [minPoint]
    for _, point in order:
        ans.append(point)
        while len(ans) > 2 and Vector(ans[-2][0] - ans[-3][0], ans[-2][1] - ans[-3][1]) \
            .cross(Vector(ans[-1][0] - ans[-2][0], ans[-1][1] - ans[-2][1])).z <= 0:
            print(f"excluding point {ans[-2]}")
            temp = ans.pop()
            ans.pop()
            ans.append(temp)

    return ans

"""
Find the centroid of a convex polygon.
@arg vertices: the polygon's vertices in order
@returns a tuple defining the convex hull
"""
def findCentroid(vertices):
    ans = [0,0]

    n = len(vertices)
    signedArea = 0

    # Do for all vertices
    for idx in range(len(vertices)):
        x0 = vertices[idx][0]
        y0 = vertices[idx][1]
        x1 = vertices[(idx + 1) % n][0]
        y1 = vertices[(idx + 1) % n][1]

        # Calculate A using shoelace formula
        A = (x0 * y1) - (x1 * y0)
        signedArea += A

        # Add weight of current vertex
        ans[0] += (x0 + x1) * A
        ans[1] += (y0 + y1) * A 
    
    signedArea *= 0.5
    ans[0] /= (6 * signedArea)
    ans[1] /= (6 * signedArea) 
    return ans


"""
Calculate the set of points needed to trace the path.
@arg points: points defining a convex polygon (not necessarily in order)
@arg gap: the gap required between traces
@returns ordered list of points defining path
"""
def buildPath(points, gap):
    # Verify number of points
    distinctPoints = set([tuple(p) for p in points])
    if (len(distinctPoints) < 3):
        raise ValueError(f"requires 3+ distinct points, given {len(distinctPoints)} points: {distinctPoints}")

    # Build convex hull & centroid
    points = convexHull(points)
    centroid = findCentroid(points)
    
    # Build segments from vertices to centroid
    segments = [LineSegment(pt[0], pt[1], centroid[0], centroid[1]) for pt in points]
    
    # Divide segments by maximum perpendicular distance from an edge to the centroid
    maxPerpDist = max([sg.distToPoint(centroid) for sg in segments])
    print("max distance:", maxPerpDist)
    numIntervals = math.ceil(maxPerpDist / gap)
    midpoints = [[sg.midpoint(1 / numIntervals * itv) for itv in range(numIntervals)] for sg in segments]

    # build final path
    path = []
    for perimeter in range(numIntervals):
        for pt in range(len(segments)):
            path.append(midpoints[pt][perimeter])
        path.append(midpoints[0][perimeter])
    path.append(centroid)

    return path

# Test the function
import random
numVertices = 4
width = round(random.uniform(0.2, 2), 2)
vertices = [[random.randint(0, 10), random.randint(0, 10)] for _ in range(numVertices)]
# vertices = [[0, 0], [4, 0], [4, 4], [0, 4]]
print("vertices: ", vertices)
print("numVertices:", numVertices, ", width:", width)
sol = buildPath(vertices, width)
print("final path:\n", sol)


# Plot results
plt.figure()
# plt.scatter([s[0] for s in sol], [s[1] for s in sol])
for idx in range(len(sol) - 1):
    plt.plot([sol[idx][0], sol[idx+1][0]], [sol[idx][1], sol[idx+1][1]], color="blue")
plt.show()

seg = LineSegment(0, 0, 2, 2)
pt = [1, 1]
# return abs((self.x2 - self.x1) * (self.y1 - point[1]) - (self.x1 - point[0]) * (self.y2 - self.y1)) / self.magnitude()
print("test:", seg.distToPoint(pt))


"""
changes to make:
determine number of intervals based on distance between centroid & perpendicular projection based on line segment


[[2, 0], [5, 5], [10, 3], [2, 7], [0, 7], [5, 0], [7, 8]]
expected:
[[2, 0], [5, 0], [10, 3], [7, 8], [0, 7]]

shallow: 
[[8, 1], [6, 6], [1, 8], [7, 5], [5, 4], [10, 0], [7, 7], [3, 9], [2, 7], [7, 8]]

vertical tricks:
[[0, 5], [0, 3], [0, 7], [6, 0], [6, 2]]
"""