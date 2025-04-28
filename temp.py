from glasses_detector import GlassesClassifier

classifier = GlassesClassifier()
classifier.process_file(
    input_path="images/SM_2.jpg",
    output_path="D:\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\output",
    format={True: "1", False: "0"},
    show=True,
)

# output_path="D:\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\runs\\detect\\predict"

# Open the file in read mode
file = open("D:\\I2I Techno Solutions\\Dlib_Static (Trial for YOLOv11)\\output\\SM_2.txt", "r")
# Read the entire content of the file
content = file.read()
# Print the content
print(content)
# Close the file
file.close()