import os
import time


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + '_%s.' + extension) % timestr
    return os.path.join(output_dir, output_string)

