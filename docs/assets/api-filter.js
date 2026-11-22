document.addEventListener("DOMContentLoaded", () => {
  const query = document.getElementById("api-query");
  if (!query) return;
  const status = document.getElementById("api-count");
  function filter() {
    const words = query.value.toLowerCase().trim().split(/\s+/);
    let count = 0;
    document.querySelectorAll(".api-group").forEach(group => {
      let groupCount = 0;
      group.querySelectorAll(".api-module").forEach(module => {
        let matches = 0;
        module.querySelectorAll("li").forEach(item => {
          const path = (module.dataset.name + "." + item.textContent).toLowerCase();
          item.hidden = !words.every(word => path.includes(word));
          if (!item.hidden) matches++;
        });
        module.hidden = matches === 0;
        groupCount += matches;
      });
      group.hidden = groupCount === 0;
      count += groupCount;
    });
    status.textContent = count ? count + " matching entries" : "No matching entries. Try a module or a shorter name.";
  }
  query.addEventListener("input", filter);
  filter();
});
