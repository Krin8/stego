import pydicom
import glob
files = glob.glob("Dataset/LungCT_Negatives/**/*.dcm", recursive=True)
success = 0
for f in files:
    try:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array
        success += 1
    except Exception as e:
        pass
print(f"Success: {success}/{len(files)}")
