from dataclasses import dataclass, field
import subprocess
from pathlib import Path
import glob
import os

@dataclass
class GitSync:
    """Klass för att hantera Git-synkronisering och CSV-filhantering.

    Attribut:
        repo_dir (Path): Sökväg till Git-repot.
        branch (str): Namn på branchen att synkronisera mot (default: 'main').
    """
    repo_dir: Path
    branch: str = "main"
    # Korrigering: Använd field(default_factory=list) för mutable standardvärden
    csv_folders: list[str] = field(default_factory=lambda: ["data", "tickers", "fundamentals", "balance", "cashflow", "income", "ranks"])
    # Notera att du behöver importera `field` från `dataclasses`

    def run_git_command(self, command: list[str]) -> str:
        """Kör ett Git-kommando och returnerar utdata.

        Args:
            command: Lista med Git-kommandot och dess argument.

        Returns:
            Utdata från kommandot som sträng.

        Raises:
            RuntimeError: Om Git-kommandot misslyckas.
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git-kommando '{' '.join(command)}' misslyckades:\n{e.stderr.strip()}")
        except FileNotFoundError:
            raise RuntimeError(f"Git-kommando '{command[0]}' hittades inte. Se till att Git är installerat och i PATH.")


    def overwrite_local_csv(self):
        """Tar bort lokala CSV-filer och gör en git pull för att hämta fjärrändringar.

        Detta säkerställer att lokala CSV-filer i de definierade mapparna skrivs
        över med de från fjärr-repot, utan att versionshantera lokala ändringar.
        """
        print("Startar synkronisering av CSV-filer...")
        try:
            print("Tar bort lokala CSV-filer...")
            total_deleted_count = 0
            for folder in self.csv_folders:
                folder_path = self.repo_dir / folder
                if not folder_path.is_dir():
                    print(f"Varning: Mappen '{folder}' hittades inte i repot. Fortsätter.")
                    continue

                deleted_count_in_folder = 0
                csv_files = glob.glob(str(folder_path / "*.csv"))
                for csv_file in csv_files:
                    try:
                        os.remove(csv_file)
                        deleted_count_in_folder += 1
                    except OSError as e:
                        print(f"Kunde inte ta bort {csv_file} i mappen '{folder}': {e}")
                
                if deleted_count_in_folder > 0:
                    print(f"Borttagna {deleted_count_in_folder} CSV-filer från mappen '{folder}'.")
                else:
                    print(f"Inga CSV-filer att ta bort i mappen '{folder}'.")
                
                total_deleted_count += deleted_count_in_folder
            
            print(f"Totalt antal borttagna CSV-filer: {total_deleted_count}")

            # Gör en git pull för att hämta fjärrändringar
            print(f"Hämtar ändringar från origin/{self.branch}...")
            pull_output = self.run_git_command(["git", "pull", "origin", self.branch])
            print("Git pull utdata:\n" + pull_output)

            print("Lokala CSV-filer har skrivits över med fjärrändringar.")
        except RuntimeError as e:
            print(f"Fel under Git-synkronisering: {e}")
        except Exception as e:
            print(f"Ett oväntat fel uppstod: {e}")


    def verify_csv_files(self):
        """Verifierar att CSV-filer finns i repot efter synkronisering.

        Skriver ut en lista över CSV-filer i de definierade mapparna som spåras av Git.
        """
        print("\nVerifierar CSV-filer...")
        try:
            patterns = [f"{folder}/*.csv" for folder in self.csv_folders]
            files = self.run_git_command(["git", "ls-files"] + patterns)

            if files:
                print("Spårade CSV-filer i repot:")
                for file in files.splitlines():
                    print(f"  {file}")
            else:
                print("Inga spårade CSV-filer hittades i de angivna mapparna i repot.")
        except RuntimeError as e:
            print(f"Fel vid verifiering: {e}")

# Användning
if __name__ == "__main__":
    repo_dir = Path.cwd()
    if not (repo_dir / ".git").is_dir():
        print(f"Fel: '{repo_dir}' är inte ett Git-repository. Vänligen kör skriptet från roten av ditt Git-repo, eller justera 'repo_dir'.")
    else:
        sync = GitSync(repo_dir=repo_dir, branch="main")
        sync.overwrite_local_csv()
        sync.verify_csv_files()