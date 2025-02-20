
path = "C:\\Users\\Asus\\Documents\\Feria\\Repositorio\\UTUAV-OD\\Data\\C_Split_80_10_10\\training\\images\\v2_006705.jpg"

def extract_id_from_path(path):
    img_name = path.split('\\')[-1].strip('.jpg').strip('v2_')
    return int(img_name)

print(extract_id_from_path(path))