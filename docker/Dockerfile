FROM gentoo/stage3:latest

RUN mkdir -p /etc/portage/binrepos.conf
COPY binrepos.conf /etc/portage/binrepos.conf/binrepos.conf

RUN emerge --sync

RUN echo 'FEATURES="-sandbox -usersandbox -ipc-sandbox -pid-sandbox -network-sandbox"' >> /etc/portage/make.conf
RUN echo 'FEATURES="${FEATURES} nostrip"' >> /etc/portage/make.conf
RUN echo 'CFLAGS="${CFLAGS} -g"' >> /etc/portage/make.conf
RUN echo 'CXXFLAGS="${CXXFLAGS} -g"' >> /etc/portage/make.conf
RUN echo 'ACCEPT_KEYWORDS="~amd64"' >> /etc/portage/make.conf

# Write .gpkg for every compiled package; fetch from remote binhosts first.
# PKGDIR is volume-mounted from the host at runtime to persist the cache.
RUN echo 'FEATURES="${FEATURES} buildpkg getbinpkg binpkg-multi-instance"' >> /etc/portage/make.conf && \
    echo 'PKGDIR="/var/cache/binpkgs"' >> /etc/portage/make.conf && \
    mkdir -p /var/cache/binpkgs

RUN emerge -q -k sys-devel/gcc:13 sys-devel/gcc:14
RUN emerge -q -k llvm-core/clang:17 llvm-core/clang:18
RUN emerge -q -k sys-devel/crossdev

RUN mkdir -p /var/db/repos/crossdev/metadata /etc/portage/repos.conf && \
    printf 'masters = gentoo\n' > /var/db/repos/crossdev/metadata/layout.conf && \
    printf '[crossdev]\nlocation = /var/db/repos/crossdev\npriority = 10\nmasters = gentoo\nauto-sync = no\n' > /etc/portage/repos.conf/crossdev.conf

RUN crossdev --stable --target aarch64-unknown-linux-gnu
RUN crossdev --stable --target mipsel-unknown-linux-gnu

RUN aarch64-unknown-linux-gnu-emerge -q sys-devel/gcc:14 || true
RUN mipsel-unknown-linux-gnu-emerge -q sys-devel/gcc:14 || true

RUN mkdir -p /workspace/output

WORKDIR /workspace
