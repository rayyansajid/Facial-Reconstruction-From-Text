import csv
import re

def update_skintones(descriptions_file, skintones_file, output_file):
    # Read the skintones from correct_skintone.csv into a dictionary
    skintone_map = {}
    with open(skintones_file, 'r') as st_file:
        reader = csv.DictReader(st_file)
        for row in reader:
            skintone_map[row['Image Name']] = row['Skin Tone']

    # Process descriptions.csv and replace skintones
    updated_rows = []
    with open(descriptions_file, 'r') as desc_file:
        reader = csv.DictReader(desc_file)
        fieldnames = reader.fieldnames
        for row in reader:
            image_name = row['Image Name']
            description = row['Description']
            
            if image_name in skintone_map:
                correct_skintone = skintone_map[image_name]
                # Replace any occurrence of "Dark", "Medium", or "Fair" in the description with the correct skintone
                updated_description = re.sub(r'\b(Dark|Medium|Fair)\b', correct_skintone, description, flags=re.IGNORECASE)
                row['Description'] = updated_description
            
            updated_rows.append(row)

    # Write the updated rows to the output file
    with open(output_file, 'w', newline='') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

# Example usage
update_skintones(r'E:\FYDP Dataset\Final_Dataset\descriptions.csv', 
                 r'C:\Users\Rayyan Sajid\OneDrive\Desktop\FYDP\Dataset\skin_tone_predictions2.csv',
                 r'E:\FYDP Dataset\Final_Dataset\updated_descriptions.csv')
