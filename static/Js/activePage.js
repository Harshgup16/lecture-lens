document.addEventListener("DOMContentLoaded", () => {
    const activePage = window.location.pathname; // Get the current page path
    const navLinks = document.querySelectorAll('#action_bar a'); // Select only nav links

    navLinks.forEach(link => {
        // Check if the link's href includes the active page
        if (link.href.includes(activePage)) {
            link.classList.add('active'); // Add 'active' class to the current link
        }
    });
});
