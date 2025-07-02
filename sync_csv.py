from dataclasses import dataclass
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
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git-kommando misslyckades: {e.stderr}")

    def overwrite_local_csv(self):
        """Tar bort lokala CSV-filer och gör en git pull för att hämta fjärrändringar.

        Detta säkerställer att lokala CSV-filer i data- och tickers-mapparna skrivs
        över med de från fjärr-repot, utan att versionshantera lokala ändringar.
        """
        try:
            # Ta bort alla lokala CSV-filer i data- och tickers-mapparna
            print("Tar bort lokala CSV-filer...")
            for folder in ["data", "tickers","fundamentals", "balance", "cashflow", "income", "ranks"]:
                csv_files = glob.glob(str(self.repo_dir / folder / "*.csv"))
                for csv_file in csv_files:
                    os.remove(csv_file)
                    print(f"Borttagen: {csv_file}")

            # Gör en git pull för att hämta fjärrändringar
            print(f"Hämtar ändringar från origin/{self.branch}...")
            self.run_git_command(["git", "pull", "origin", self.branch])

            print("Lokala CSV-filer har skrivits över med fjärrändringar.")
        except (OSError, RuntimeError) as e:
            print(f"Fel: {e}")

    def verify_csv_files(self):
        """Verifierar att CSV-filer finns i repot efter synkronisering.

        Skriver ut en lista över CSV-filer i data- och tickers-mapparna som spåras av Git.
        """
        try:
            files = self.run_git_command(["git", "ls-files", "data/*.csv", "tickers/*.csv"])
            if files:
                print("Spårade CSV-filer i data- och tickers-mapparna:")
                for file in files.splitlines():
                    print(f"  {file}")
            else:
                print("Inga CSV-filer hittades i data- eller tickers-mapparna i repot.")
        except RuntimeError as e:
            print(f"Fel vid verifiering: {e}")

# Användning
if __name__ == "__main__":
    repo_dir = Path.cwd()  # Använd aktuell katalog som repo
    sync = GitSync(repo_dir=repo_dir, branch="main")

    # Skriv över lokala CSV-filer och hämta fjärrändringar
    sync.overwrite_local_csv()

    # Verifiera att CSV-filer finns
    sync.verify_csv_files()