"""Append one finding's verified result and changed source line ranges."""
import pathlib, re, subprocess, sys
finding, status, verification, remaining = sys.argv[1:]
diff = subprocess.check_output(['git', 'diff', '--unified=0', '--', 'src'], text=True)
locations = []
path = ''
for line in diff.splitlines():
    if line.startswith('+++ b/'): path = line[6:]
    match = re.match(r'@@ .* \+(\d+)(?:,(\d+))? @@', line)
    if match: locations.append(f'{path}:{match[1]}' + (f'–{int(match[1])+int(match[2])-1}' if match[2] and int(match[2]) > 1 else ''))
p = pathlib.Path('audit/FIXES.md')
s = p.read_text()
row = ' | '.join([finding, status, ', '.join(locations) or 'No production edits', verification, remaining])
s = s.replace('\n## New observations', '\n' + row + '\n\n## New observations')
p.write_text(s)
