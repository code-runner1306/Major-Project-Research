document.querySelectorAll('[data-decay-lab]').forEach(lab=>{
  const alpha=lab.querySelector('[name=alpha]'),steps=lab.querySelector('[name=steps]');
  const update=()=>{
    const a=Number(alpha.value),n=Number(steps.value),retained=100*Math.pow(a,n);
    lab.querySelector('[data-alpha]').textContent=a.toFixed(3);
    lab.querySelector('[data-steps]').textContent=n;
    lab.querySelector('[data-retained]').textContent=retained.toFixed(1)+'%';
    lab.querySelector('.bar span').style.width=retained+'%';
  };
  alpha.addEventListener('input',update);steps.addEventListener('input',update);update();
});
document.querySelectorAll('[data-question]').forEach(question=>{
  question.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{
    question.querySelector('.feedback').textContent=button.dataset.correct==='true'
      ?question.dataset.success:question.dataset.retry;
  }));
});
