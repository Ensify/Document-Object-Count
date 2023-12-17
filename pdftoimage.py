import fitz
import sys
filename = sys.argv[1]

try:
    doc = fitz.open(filename)
    page = doc[0]
    pix = page.get_pixmap(dpi=360)  
    pix.save("image.png") 
    print("Images saved at image.png")

except:
    print("Invalid pdf")