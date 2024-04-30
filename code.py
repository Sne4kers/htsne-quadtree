import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tree
import numpy as np

# Random testing
my_tree = tree.PyInfinityQuadTree()
print(my_tree.get_nodes())

print(dir(my_tree))


# generate points
n = 500

points = []
for i in range(n):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    while x*x + y*y > 1:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
    points.append((x, y))

# make the tree
my_tree.build_tree(points, -1, -1, 1, 1)

print(my_tree.get_nodes())

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

for i in my_tree.get_nodes():
    print("something")
    if i["is_leaf"]:
        color = "b"
    else:
        color = "r"
    rect = patches.Rectangle((i["min_bounds"][0], i["min_bounds"][1]), i["max_bounds"][0] - i["min_bounds"][0], i["max_bounds"][1] - i["min_bounds"][1], linewidth=0.25, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

plt.savefig('foo.png')