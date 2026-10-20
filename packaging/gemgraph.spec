Name:           gemgraph
Version:        0.1.0
Release:        1%{?dist}
Summary:        Clean-Room Scientific Data Analysis & Curve Fitting

License:        GPL-3.0-or-later
URL:            https://github.com/nicolauda/gemgraph
BuildArch:      x86_64

# User-space rpmbuild packaging spec for frozen PyInstaller bundle.
# Enforces No-Sudo-Strict-Local policy (pure user-space packaging).
# Build command:
#   rpmbuild -bb --define "_topdir $(pwd)/rpmbuild" packaging/gemgraph.spec

AutoReqProv:    no
%define __spec_install_post %{nil}
%define _enable_debug_packages 0
%define debug_package %{nil}
%{!?_projectroot: %define _projectroot %(pwd)}

%description
GemGraph is an open-source, clean-room scientific data analysis, curve fitting,
and graphing software application designed as a modern alternative to
GraphPad Prism. Features 8 table paradigms, Levenberg-Marquardt nonlinear
regression, ROUT outlier rejection, ANOVA with Dunnett/Tukey post-hoc tests,
and publication-grade vector graphing.

%prep
# Pre-compiled via PyInstaller; no compilation prep required.

%build
# Pre-compiled via PyInstaller in dist/GemGraph.

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}/opt/gemgraph
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_datadir}/applications

# Locate and copy frozen PyInstaller distribution from dist/GemGraph
if [ -d "%{_projectroot}/dist/GemGraph" ]; then
    cp -a "%{_projectroot}/dist/GemGraph/"* %{buildroot}/opt/gemgraph/
elif [ -d "%{_sourcedir}/GemGraph" ]; then
    cp -a "%{_sourcedir}/GemGraph/"* %{buildroot}/opt/gemgraph/
elif [ -d "dist/GemGraph" ]; then
    cp -a "dist/GemGraph/"* %{buildroot}/opt/gemgraph/
else
    echo "ERROR: dist/GemGraph not found. Build PyInstaller bundle first." >&2
    exit 1
fi

# Install /usr/bin launcher
cat << 'EOF' > %{buildroot}%{_bindir}/gemgraph
#!/bin/sh
exec /opt/gemgraph/GemGraph "$@"
EOF
chmod +x %{buildroot}%{_bindir}/gemgraph

# Install desktop entry
if [ -f "%{_projectroot}/packaging/gemgraph.desktop" ]; then
    cp -a "%{_projectroot}/packaging/gemgraph.desktop" %{buildroot}%{_datadir}/applications/gemgraph.desktop
else
    cat << 'EOF' > %{buildroot}%{_datadir}/applications/gemgraph.desktop
[Desktop Entry]
Type=Application
Name=GemGraph
Comment=Scientific Data Analysis, Non-linear Regression & Graphing
Exec=/usr/bin/gemgraph
Icon=utilities-terminal
Terminal=false
Categories=Science;Math;Education;
EOF
fi

%files
/opt/gemgraph
%{_bindir}/gemgraph
%{_datadir}/applications/gemgraph.desktop

%changelog
* Mon Oct 19 2026 GemGraph Team <feedback@gemgraph.org> - 0.1.0-1
- Initial v0.1.0-alpha release
