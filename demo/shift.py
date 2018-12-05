import os

for videoDir in os.listdir("out"):
    for filename in os.listdir("out/" + videoDir):
        path = "out/" + videoDir + "/" + filename
        if not filename.endswith(".png"):
            os.remove(path)
            break

        no_extension = filename.split(".")[0]
        new_name = str(int(no_extension) - 1)
        os.rename(path, "out/" + videoDir + "/" + new_name + ".png")
