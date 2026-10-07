# Downloads the Silesia and Canterbury corpora into testdata/corpus/ (git-ignored).
# Usage: scripts\fetch_corpus.ps1    then: gpucomp bench --corpus testdata\corpus\silesia
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "testdata/corpus"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    function Fetch($name, $url, $kind) {
        $out = Join-Path $dest $name
        if ((Test-Path $out) -and (Get-ChildItem $out)) { Write-Host "${name}: already present"; return }
        Write-Host "${name}: downloading $url"
        $archive = Join-Path $tmp "$name.$kind"
        Invoke-WebRequest -Uri $url -OutFile $archive
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        if ($kind -eq "zip") { Expand-Archive -Path $archive -DestinationPath $out -Force }
        else { tar -xzf $archive -C $out }
        Write-Host "${name}: $((Get-ChildItem $out).Count) files"
    }
    Fetch "silesia" "https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip" "zip"
    Fetch "canterbury" "https://corpus.canterbury.ac.nz/resources/cantrbry.tar.gz" "tar.gz"
} finally {
    Remove-Item -Recurse -Force $tmp
}
