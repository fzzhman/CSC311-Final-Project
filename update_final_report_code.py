import os
import shutil
import glob

all_code = glob.glob("./code/**/*.py")
final_report_dir = "./final_report"

for code in all_code:
    dst = os.path.join(final_report_dir, os.path.basename(code))
    if os.path.isfile(dst):
        os.remove(dst)
    shutil.copy(code, dst)