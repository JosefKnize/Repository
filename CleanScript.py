import os, shutil
if __name__ == "__main__":

    datasetPath = ".\..\data03\\"
    directories = os.listdir(datasetPath)

    for folder in directories:
        if folder != "MachineData":
            for subFolder in ["Aligned"]:
            # for subFolder in ["Aligned", "Jsons", "Processed"]:
                subFolderPath = os.path.join(folder, subFolder)
                subFolderPath = os.path.join(datasetPath, subFolderPath)
                for filename in os.listdir(subFolderPath):
                    file_path = os.path.join(subFolderPath, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))  
        if folder == "MachineData":
            for subFolder in ["ValidationSet","TrainSet"]:
                subFolderPath = os.path.join(folder, subFolder)
                subFolderPath = os.path.join(datasetPath, subFolderPath)
                for filename in os.listdir(subFolderPath):
                    file_path = os.path.join(subFolderPath, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))  
            