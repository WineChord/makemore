import os
import requests
import re
from typing import List, Tuple, Dict

class HolmesDownloader:
    """A class to download and process Sherlock Holmes stories."""
    
    def __init__(self) -> None:
        # 定义所有福尔摩斯小说的URL和标题。
        self.books: List[Tuple[str, str]] = [
            ("A Study in Scarlet", 
             "https://www.gutenberg.org/files/244/244-0.txt"),
            ("The Sign of the Four", 
             "https://www.gutenberg.org/files/2097/2097-0.txt"),
            ("The Hound of the Baskervilles", 
             "https://www.gutenberg.org/files/2852/2852-0.txt"),
            ("The Valley of Fear", 
             "https://www.gutenberg.org/files/3289/3289-0.txt"),
            ("The Adventures of Sherlock Holmes", 
             "https://www.gutenberg.org/files/1661/1661-0.txt"),
            ("The Memoirs of Sherlock Holmes", 
             "https://www.gutenberg.org/files/834/834-0.txt"),
            ("The Return of Sherlock Holmes", 
             "https://www.gutenberg.org/files/108/108-0.txt"),
            ("His Last Bow", 
             "https://www.gutenberg.org/files/2350/2350-0.txt"),
        ]
        
        # 创建下载目录。
        self.raw_dir = "raw_holmes"
        self.ensure_directory(self.raw_dir)
        
    def ensure_directory(self, directory: str) -> None:
        """确保目录存在，如果不存在则创建。"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    def download_books(self) -> None:
        """下载所有书籍。"""
        for title, url in self.books:
            filename = os.path.join(self.raw_dir, f"{title}.txt")
            if not os.path.exists(filename):
                print(f"Downloading {title}...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                except Exception as e:
                    print(f"Error downloading {title}: {e}")
            else:
                print(f"{title} already exists.")
                
    def clean_text(self, text: str) -> str:
        """Clean the text content by removing Gutenberg headers and footers.
        
        Args:
            text: The raw text content to be cleaned.
            
        Returns:
            The cleaned text content.
        """
        # Find the start and end positions of the actual content.
        start_marker = "*** START OF"
        end_marker = "*** END OF"
        
        try:
            # Find the content boundaries.
            start_pos = text.index(start_marker)
            start_pos = text.index("***", start_pos + len(start_marker)) + 3
            end_pos = text.index(end_marker)
            
            # Extract the main content.
            text = text[start_pos:end_pos].strip()
            
            # Process the lines more efficiently.
            lines = []
            empty_line_count = 0
            
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    empty_line_count += 1
                    if empty_line_count <= 2:  # Keep at most 2 consecutive empty lines.
                        lines.append('')
                else:
                    empty_line_count = 0
                    lines.append(line)
            
            return '\n'.join(lines)
            
        except ValueError as e:
            # If markers not found, return cleaned version of original text.
            print(f"Warning: Could not find content markers, falling back to regex.")
            # Fallback to regex for problematic files.
            text = re.sub(r'[\s\S]*?\*\*\* START OF .+? \*\*\*', '', text)
            text = re.sub(r'\*\*\* END OF .+?\*\*\*[\s\S]*', '', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return '\n'.join(line.strip() for line in text.split('\n'))
    
    def merge_books(self) -> None:
        """Merge all downloaded books into a single file."""
        print("Merging books...")
        
        # Check if all files exist before processing.
        missing_files = []
        for title, _ in self.books:
            filename = os.path.join(self.raw_dir, f"{title}.txt")
            if not os.path.exists(filename):
                missing_files.append(title)
        
        if missing_files:
            print("Error: The following books are missing:")
            for title in missing_files:
                print(f"- {title}")
            return
        
        def generate_content():
            """Generator function to yield content line by line."""
            total_books = len(self.books)
            for idx, (title, _) in enumerate(self.books, 1):
                filename = os.path.join(self.raw_dir, f"{title}.txt")
                print(f"Processing {title} ({idx}/{total_books})...")
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        # Yield the separator lines.
                        yield f"\n\n{'='*80}\n"
                        yield f"{title.upper()}\n"
                        yield f"{'='*80}\n\n"
                        
                        # Process and yield the content.
                        content = f.read()
                        cleaned_content = self.clean_text(content)
                        yield cleaned_content
                        yield "\n"
                        
                    print(f"Successfully processed {title}.")
                except Exception as e:
                    print(f"Error processing {title}: {str(e)}")
                    return
        
        # Write the merged file using a generator.
        try:
            with open('complete_holmes.txt', 'w', encoding='utf-8', 
                     buffering=1024*1024) as f:  # 1MB buffer
                f.writelines(generate_content())
            print("Complete Holmes collection has been created as 'complete_holmes.txt'.")
        except Exception as e:
            print(f"Error writing complete_holmes.txt: {str(e)}")

def main() -> None:
    """主函数。"""
    downloader = HolmesDownloader()
    downloader.download_books()
    downloader.merge_books()
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
  