# Gentoo-based build image for bin2vec
# Provides multiple GCC and Clang versions, plus cross-compilation toolchains
#
# Binary package cache (binpkg)
# ------------------------------
# PKGDIR=/var/cache/binpkgs is where Portage writes compiled binary packages
# (.gpkg tarballs) when FEATURES=buildpkg is set.  Mount a host directory
# there at `docker run` time so that packages built during image construction
# AND packages built during `bin2vec build` runs are all cached on the host
# and survive container restarts / image rebuilds.
#
# At `docker build` time we mount nothing (BuildKit has no persistent volumes),
# so the first build still compiles from source.  On the SECOND build (or when
# the same image is used for bin2vec build runs) every already-compiled package
# is installed in seconds from the .gpkg file instead of recompiled.
#
# Usage on the host:
#   mkdir -p data/binpkgs
#   docker build -t bin2vec-gentoo docker/
#   # Then start containers with -v $PWD/data/binpkgs:/var/cache/binpkgs

FROM gentoo/stage3:latest

# Sync portage tree
RUN emerge --sync

# ---------------------------------------------------------------------------
# Global Portage configuration
# ---------------------------------------------------------------------------

# Disable sandbox features (not supported in Docker)
RUN echo 'FEATURES="-sandbox -usersandbox -ipc-sandbox -pid-sandbox -network-sandbox"' >> /etc/portage/make.conf

# Keep debug info in binaries (nostrip) for DWARF extraction
RUN echo 'FEATURES="${FEATURES} nostrip"' >> /etc/portage/make.conf
RUN echo 'CFLAGS="${CFLAGS} -g"' >> /etc/portage/make.conf
RUN echo 'CXXFLAGS="${CXXFLAGS} -g"' >> /etc/portage/make.conf

# Accept all keywords for latest packages
RUN echo 'ACCEPT_KEYWORDS="~amd64"' >> /etc/portage/make.conf

# ---------------------------------------------------------------------------
# Binary package cache configuration
# ---------------------------------------------------------------------------
# buildpkg   – write a binary package (.gpkg) for every package emerged
# binpkg-multi-instance – allow multiple versions/slots of the same package
# getbinpkg  – prefer installing from binary packages when available
#
# PKGDIR is the on-disk location for those .gpkg files.
# We set it to /var/cache/binpkgs so it can be volume-mounted from the host.
RUN echo 'FEATURES="${FEATURES} buildpkg binpkg-multi-instance"' >> /etc/portage/make.conf && \
    echo 'PKGDIR="/var/cache/binpkgs"' >> /etc/portage/make.conf && \
    mkdir -p /var/cache/binpkgs

# ---------------------------------------------------------------------------
# Install toolchains (write binpkgs as we go for future cache hits)
# ---------------------------------------------------------------------------

# Install multiple GCC slots
# -k / --usepkg: use a binary package if one already exists in PKGDIR
RUN emerge -q -k sys-devel/gcc:13 sys-devel/gcc:14

# Install multiple Clang/LLVM slots
RUN emerge -q -k llvm-core/clang:17 llvm-core/clang:18

# Install crossdev for cross-compilation toolchains
RUN emerge -q -k sys-devel/crossdev

# Set up crossdev overlay
RUN mkdir -p /var/db/repos/crossdev/metadata /etc/portage/repos.conf && \
    printf 'masters = gentoo\n' > /var/db/repos/crossdev/metadata/layout.conf && \
    printf '[crossdev]\nlocation = /var/db/repos/crossdev\npriority = 10\nmasters = gentoo\nauto-sync = no\n' > /etc/portage/repos.conf/crossdev.conf

# Create cross-compilation toolchains
# crossdev itself compiles cross-gcc/glibc; those are also cached as binpkgs
RUN crossdev --stable --target aarch64-unknown-linux-gnu
RUN crossdev --stable --target mipsel-unknown-linux-gnu

# Install additional GCC slots for cross targets
RUN aarch64-unknown-linux-gnu-emerge -q sys-devel/gcc:14 || true
RUN mipsel-unknown-linux-gnu-emerge -q sys-devel/gcc:14 || true

# Create workspace
RUN mkdir -p /workspace/output

WORKDIR /workspace
