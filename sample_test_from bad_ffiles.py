import os

for root, _, files in os.walk("."):
    for f in files:
        path = os.path.join(root, f)
        try:
            with open(path, encoding="utf-8") as file:
                file.read()
        except Exception as e:
            print("Problem file:", path)