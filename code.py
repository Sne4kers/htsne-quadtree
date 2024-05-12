import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tree
import qt_tree
import numpy as np

# Random testing
experiments = 1

a = 0
b = 0

# generate points
n = 100

points = []
for i in range(n):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    while x*x + y*y > 1:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
    points.append((x, y))

    # Add radially pposite node
    #points.append((-x, -y))

print("POINTS ", points)

# make the tree
my_tree = tree.PyInfinityQuadTree(points)
qt_tree = qt_tree._QuadTree(2, 50)

points_extra = np.array(points)
print(points_extra.shape)
qt_tree.build_tree(points_extra)
print(dir(my_tree))
print(my_tree.get_nodes())

print("-------------")
print(my_tree.summarize(np.array([0.0, 0.0]), points_extra, 0.5)[0])
print(np.asarray(my_tree.summarize(np.array([0.0, 0.0]), points_extra, 0.5)[1]))
print("-------------")
print(qt_tree._py_summarize(np.array([0.0, 0.0]), points_extra, 0.5)[0])
print(np.asarray(qt_tree._py_summarize(np.array([0.0, 0.0]), points_extra, 0.5)[1]))
print("-------------")

print(qt_tree.get_root()["barycenter"])

# Prepare  the plot
plt.xlim(-1, 1)
plt.ylim(-1, 1)
#plt.figure(figsize=(12, 12))
#plt.gca().set_aspect('equal')

fig, ax = plt.subplots()

fig.set_figheight(15)
fig.set_figwidth(15)


pdisk = patches.Circle((0, 0), 1, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(pdisk)

for p in points:
    ax.scatter(p[0], p[1], color="g", s=5)

selected_depth = 3

for i in my_tree.get_nodes():
    if i["is_leaf"]:
        color = "r"
    else:
        color = "b"
        if i["depth"] == 0:
            print(i["barycenter"]["x"], i["barycenter"]["y"])
    ax.scatter(i["barycenter"]["x"], i["barycenter"]["y"], c="orange", alpha=0.5)
    rect = patches.Rectangle((i["min_bounds"]["x"], i["min_bounds"]["y"]), i["max_bounds"]["x"] - i["min_bounds"]["x"], i["max_bounds"]["y"] - i["min_bounds"]["y"], linewidth=0.25, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

plt.savefig('foo.png')
# a /= experiments
# b /= experiments
# print("FINAL RESULTS ", a, b)