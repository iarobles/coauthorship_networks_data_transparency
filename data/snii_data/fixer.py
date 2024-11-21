import pandas as pd

# Ruta del archivo CSV original
input_file = './data/snii_data/salud_ego_included_coco.csv'

# Leer el archivo CSV
df = pd.read_csv(input_file)

# Eliminar las columnas 'cvu', 'names' y 'surnames'
df_anonymized = df.drop(columns=['cvu'])

# Guardar el nuevo archivo CSV con el sufijo '_anonymized'
output_file = input_file.replace('.csv', '_anonymized.csv')
df_anonymized.to_csv(output_file, index=False)

print(f'Archivo guardado como: {output_file}')
