# Gentoo-based build image for bin2vec
# Provides multiple GCC and Clang versions, plus cross-compilation toolchains
FROM gentoo/stage3:latest

# Sync portage tree
RUN emerge --sync

# Disable sandbox features (not supported in Docker)
RUN echo 'FEATURES="-sandbox -usersandbox -ipc-sandbox -pid-sandbox -network-sandbox"' >> /etc/portage/make.conf

# Keep debug info in binaries (nostrip) for DWARF extraction
RUN echo 'FEATURES="${FEATURES} nostrip"' >> /etc/portage/make.conf
RUN echo 'CFLAGS="${CFLAGS} -g"' >> /etc/portage/make.conf
RUN echo 'CXXFLAGS="${CXXFLAGS} -g"' >> /etc/portage/make.conf

# Accept all keywords for latest packages
RUN echo 'ACCEPT_KEYWORDS="~amd64"' >> /etc/portage/make.conf

# Install multiple GCC slots
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
RUN crossdev --stable --target aarch64-unknown-linux-gnu
RUN crossdev --stable --target mipsel-unknown-linux-gnu

# Install additional GCC slots for cross targets
RUN aarch64-unknown-linux-gnu-emerge -q sys-devel/gcc:14 || true
RUN mipsel-unknown-linux-gnu-emerge -q sys-devel/gcc:14 || true

# Create workspace
RUN mkdir -p /workspace/output

WORKDIR /workspace
