"""Read only selected upstream pipeline source; no downloads saved."""
import sys, urllib.request
REV = 'eb4d6b521740bd4265681b353547f2ffca65d673'
for spec in sys.argv[1:]:
    path, _, ranges = spec.partition(':')
    text = urllib.request.urlopen('https://raw.githubusercontent.com/mhamilton723/STEGO/'+REV+'/'+path, timeout=20).read().decode()
    spans = [tuple(map(int, r.split('-'))) for r in ranges.split(',')] if ranges else [(1,10000)]
    print('\nREFERENCE', path, REV)
    for i,line in enumerate(text.splitlines(),1):
        if any(a <= i <= b for a,b in spans): print(f'{i}: {line}')
