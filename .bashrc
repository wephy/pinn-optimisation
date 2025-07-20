gitpush() {
    git add -A
    git commit -m "$*"
    git push
}
alias gp=gitpush

alias python=python3
