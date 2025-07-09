import os
import json
import glob
import numpy as np
from xml.etree import ElementTree as ET
import torch
import random

from pc.plot_gen.plot_utils import extract_vertical_axes_coords, safe_join

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class LineCoordinateExtractor:
    def __init__(self, main_dir, output_file="alldata.json"):
        self.main_dir = main_dir
        self.output_file = safe_join(self.main_dir, output_file)

    @staticmethod
    def parse_path_data(d):
        points = []
        current_point = (0, 0)
        lines = []

        commands = []
        temp_command = ""
        for char in d:
            if char.isalpha():
                if temp_command:
                    commands.append(temp_command.strip())
                temp_command = char
            else:
                temp_command += char
        if temp_command:
            commands.append(temp_command.strip())

        for command in commands:
            cmd = command[0]
            if cmd in 'ML':
                coords = command[1:].split(' ')
                for coord in coords:
                    if coord:
                        x, y = map(float, coord.split(','))
                        new_point = (round(x, 2), round(y + 11, 2))
                        if cmd == 'L':
                            lines.append((current_point, new_point))
                        points.append(new_point)
                        current_point = new_point
            elif cmd == 'H':
                x = float(command[1:])
                new_point = (x, current_point[1])
                lines.append((current_point, new_point))
                points.append(new_point)
                current_point = new_point
            elif cmd == 'V':
                y = float(command[1:])
                new_point = (current_point[0], y + 11)
                lines.append((current_point, new_point))
                points.append(new_point)
                current_point = new_point
            elif cmd == 'Z':
                if points:
                    lines.append((current_point, points[0]))
                    current_point = points[0]

        return points, lines

    def extract_all(self):
        svg_files = sorted(glob.glob(os.path.join(self.main_dir, "*.svg")))
        all_data = []

        for svg_file_path in svg_files:
            vertical_axes = extract_vertical_axes_coords(svg_file_path)
            tree = ET.parse(svg_file_path)
            root = tree.getroot()
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            ET.register_namespace('', namespace['svg'])

            elements = []
            for elem in root.findall('.//svg:path', namespace):
                if elem.attrib.get('aria-roledescription') == 'tick':
                    continue
                path_data = elem.attrib.get('d', None)
                if path_data:
                    elements.append(path_data)

            points_set = set()
            lines = []
            for element in elements:
                extracted_points, extracted_lines = self.parse_path_data(element)
                points_set.update(extracted_points)
                lines.extend(extracted_lines)

            points = list(points_set)

            for i in range(len(vertical_axes) - 1):
                left_x = vertical_axes[i]
                right_x = vertical_axes[i + 1]

                region_lines = [
                    [round(start[0] - left_x, 2), round(start[1], 2),
                     round(end[0] - left_x, 2), round(end[1], 2)]
                    for start, end in lines
                    if left_x <= start[0] <= right_x and left_x <= end[0] <= right_x
                ]

                region_filename = os.path.basename(svg_file_path).replace('.svg', '')
                region_data = {
                    "filename": f"{region_filename}_crop_{i + 1}.png",
                    "lines": region_lines
                }

                all_data.append(region_data)

        with open(self.output_file, 'w') as json_file:
            json.dump(all_data, json_file, indent=4)

        return self.output_file
