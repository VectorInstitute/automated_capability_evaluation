import os
import shutil

def delete_extra_dirs(folder_a, folder_b, dry_run=True):
    """
    Delete directories that are in folder_a but not in folder_b.
    
    Args:
        folder_a (str): Path to folder A.
        folder_b (str): Path to folder B.
        dry_run (bool): If True, only print what would be deleted. 
                        Set to False to actually delete.
    """
    # Get directories in folder A
    dirs_a = {d for d in os.listdir(folder_a) if os.path.isdir(os.path.join(folder_a, d))}
    # Get directories in folder B
    dirs_b = {d for d in os.listdir(folder_b) if os.path.isdir(os.path.join(folder_b, d))}
    
    # Directories to remove (in A but not in B)
    extra_dirs = dirs_a - dirs_b
    
    for d in extra_dirs:
        path_to_remove = os.path.join(folder_a, d)
        if dry_run:
            print(f"[DRY RUN] Would delete: {path_to_remove}")
        else:
            print(f"Deleting: {path_to_remove}")
            shutil.rmtree(path_to_remove)

# Example usage:
folder_a = "/h/fkohankh/ace_logs/capabilities_o3-mini_farnaz/math"
folder_b = "/h/fkohankh/ace_logs/capabilities_results_o3-mini_farnaz/o3-mini/math"

delete_extra_dirs(folder_a, folder_b, dry_run=False)  # Preview
# delete_extra_dirs(folder_a, folder_b, dry_run=False)  # Actually delete
