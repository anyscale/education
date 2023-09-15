# Anyscale Endpoints Lab

This lab is an opportunity to familiarize yourself with
* Anyscale Endpoints -- a service similar the OpenAI API ... except all of the models are open
* LLM apps in general
* How LLM apps change based on the models they use

If you're new to LLM applications, focus on the intro level activity. If you've worked with LLM apps before, you may have time to try the additional activities.

## Intro level activity

The main activity in this lab is to duplicate the Intro_LLM_Apps notebook, and 
* switch the "backend" to Anyscale Endpoints by changing the API key and base URL as described at https://app.endpoints.anyscale.com/
    * during Ray Summit, you can use a private Anyscale Endpoints instance by using `https://ray-summit-training-jrvwy.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata-staging.com/v1` as the API base URL
* switch the model to "meta-llama/Llama-2-70b-chat-hf" (the 70B Llama 2 chat-tuned model)
* the open LLama-2-70b-chat model sometimes produces much longer responses (that also take more time)
    * `max_tokens=100` to the `create()` call to get shorter faster answers
    * you may need to allow more tokens to get properly formatted answers to the email (tool) questions
Feel free to delete the extra explanatory cells, so that you can focus on keeping the code cells close together

Notice that some of the model behavior is a bit different from the GPT-3.5-Turbo version

## Intermediate level activity

Try to change the prompts in order to get this model to achieve the same level of functionality as GPT-3.5-Turbo

## Advanced activity

Alter the email tool/agent functionality to also include the destination email address (since sending a real email would of course require the target address).

Hint: this will require altering the "database" as well as the prompts
